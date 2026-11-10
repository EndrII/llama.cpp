import collections
import csv
import json
from pathlib import Path
import statistics
import sys
import re

root = Path(sys.argv[1])

def union_ns(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(b-a for a,b in merged)

def short_name(name):
    name = name.replace('(anonymous namespace)::','').removeprefix('void ')
    return name.split('(')[0] if '<' not in name else name[:name.find('>(')+1] if '>(' in name else name.split('(')[0]

def family(row, state):
    name = row['Kernel_Name']
    if '::k_mv_rot_rows<' in name:
        if state.get('last_mv') == 'ffn_gate_up':
            label = 'FFN down INT4 matvec + rotation'
        else:
            label = 'GDN output INT4 matvec + rotation' if state.get('layer') == 'gdn' else 'Attention output INT4 matvec + rotation'
        state['last_mv'] = 'rot'
        return label
    if '::k_mv<' in name:
        if 'k_mv<1, true,' in name:
            state['last_mv'] = 'ffn_gate_up'
            return 'FFN gate + up INT4 matvec'
        if 'k_mv<1, false, true,' in name:
            if state.get('last_mv') == 'ffn_gate_up':
                label = 'FFN down INT4 matvec + rotation'
            else:
                label = 'GDN output INT4 matvec + rotation' if state.get('layer') == 'gdn' else 'Attention output INT4 matvec + rotation'
            state['last_mv'] = 'rot'
            return label
        if 'k_mv<1, false, false, true, true,' in name:
            state['layer'] = 'gdn'
            state['last_mv'] = 'qkvz'
            return 'GDN QKV + z INT4 matvec'
        if re.search(r'k_mv<1,\s*false,\s*false,\s*false,\s*false,\s*true(?:,|>)',name):
            state['layer'] = 'attention'
            state['last_mv'] = 'qkv'
            return 'Attention QKV INT4 matvec'
        return 'Other INT4 matvec'
    if '::k_gemm' in name:
        variant = re.search(r'k_gemm(?:8w|8)?<([^>]+)>',name).group(1)
        waves = int(variant.split(',')[-1].strip())
        features = int(row['Grid_Size_X']) // waves
        previous = state.get('last_gemm_features')
        state['last_gemm_features'] = features
        if features == 17408: return 'FFN gate + up INT4 GEMMs'
        if features == 10240:
            state['prefill_layer'] = 'gdn'
            return 'GDN QKV INT4 GEMM'
        if features == 12288:
            state['prefill_layer'] = 'attention'
            return 'Attention Q INT4 GEMM'
        if features == 6144: return 'GDN z INT4 GEMM'
        if features == 1024: return 'Attention K/V INT4 GEMMs'
        if features == 5120:
            if previous == 17408: return 'FFN down INT4 GEMM'
            return 'GDN output INT4 GEMM' if state.get('prefill_layer') == 'gdn' else 'Attention output INT4 GEMM'
        return 'Other INT4 GEMM'
    if 'gdn_chunk_prep' in name: return 'GDN chunk preparation'
    if 'gdn_chunk_scan' in name: return 'GDN chunk scan'
    if 'gated_delta_net' in name: return 'GDN recurrent update'
    if 'k_rmsnorm_rot' in name: return 'RMS norm + rotation + activation packing'
    if 'k_rot_pack' in name or 'k_rot_q8x' in name: return 'Rotation + activation packing'
    if 'flash_attn' in name: return 'Flash attention'
    if 'attn_prep' in name: return 'Attention preparation'
    if name.startswith('void mul_mat_vec_q<'): return 'Vocabulary output head'
    if 'mul_mat_vec_f<' in name: return 'BF16 alpha/beta matvecs'
    if 'Cijk_' in name: return 'hipBLAS BF16 GEMM'
    if 'rms_norm' in name: return 'Other RMS norms'
    if 'rocclr' in name: return 'Runtime copy/fill kernels'
    if 'cpy_' in name: return 'Tensor copies'
    if 'convert_' in name: return 'Type conversion'
    if 'norm' in name: return 'Other normalization'
    if 'concat' in name: return 'Concat'
    if 'conv' in name: return 'SSM convolution'
    if 'rope' in name: return 'RoPE'
    if 'k_bin_bcast' in name or 'unary_' in name or 'scale_f32' in name: return 'Elementwise operations'
    return short_name(name)

result = {}
for path in sorted(root.glob('*/run_kernel_trace.csv')):
    test = path.parent.name
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row['start'] = int(row['Start_Timestamp'])
        row['end'] = int(row['End_Timestamp'])
    rows.sort(key=lambda row: row['start'])
    heads = [i for i,r in enumerate(rows) if r['Kernel_Name'].startswith('void mul_mat_vec_q<')]
    count = 100 if test == 'decode' else 2
    assert len(heads) > count, (test,len(heads))
    windows = [(rows[a]['end'],rows[b]['end'],rows[a+1:b+1]) for a,b in zip(heads[-count-1:-1],heads[-count:])]
    groups = collections.defaultdict(list)
    kernels = collections.defaultdict(list)
    selected = []
    for start,end,part in windows:
        state = {}
        for row in part:
            label = family(row,state)
            groups[label].append(row)
            key = (short_name(row['Kernel_Name']),row['Grid_Size_X'],row['Grid_Size_Y'],row['Grid_Size_Z'])
            kernels[key].append(row)
        selected += part
    total_ns = sum(r['end']-r['start'] for r in selected)
    wall_ns = sum(end-start for start,end,part in windows)
    kernel_union = sum(union_ns((r['start'],r['end']) for r in part) for start,end,part in windows)
    copies = []
    copyfile = path.parent/'run_memory_copy_trace.csv'
    if copyfile.exists():
        for row in csv.DictReader(copyfile.open()):
            a,b=int(row['Start_Timestamp']),int(row['End_Timestamp'])
            if any(a>=start and b<=end for start,end,part in windows):copies.append((a,b))
    traced_union = sum(union_ns([(r['start'],r['end']) for r in part]+[(a,b) for a,b in copies if a>=start and b<=end]) for start,end,part in windows)
    def summary(label,rs):
        durations=[(r['end']-r['start'])/1e6 for r in rs]
        return {'name':label,'calls_per_unit':len(rs)/count,'total_ms_per_unit':sum(durations)/count,
                'mean_us_per_call':1000*statistics.mean(durations),'median_us_per_call':1000*statistics.median(durations),
                'gpu_time_pct':sum(durations)*1e6/total_ns*100,'wall_pct':sum(durations)*1e6/wall_ns*100,
                'max_scratch_bytes':max(int(r['Scratch_Size']) for r in rs),
                'max_vgpr':max(int(r['VGPR_Count']) for r in rs)}
    g=sorted([summary(k,v) for k,v in groups.items()],key=lambda r:-r['total_ms_per_unit'])
    k=sorted([dict(summary(key[0],rs),grid_x=key[1],grid_y=key[2],grid_z=key[3],symbol=rs[0]['Kernel_Name']) for key,rs in kernels.items()],key=lambda r:-r['total_ms_per_unit'])
    data={'units':count,'unit':'token' if test=='decode' else 'prefill pass',
          'wall_ms_per_unit':wall_ns/count/1e6,'gpu_kernel_sum_ms_per_unit':total_ns/count/1e6,
          'kernel_union_ms_per_unit':kernel_union/count/1e6,'kernel_and_copy_union_ms_per_unit':traced_union/count/1e6,
          'uncovered_ms_per_unit':(wall_ns-traced_union)/count/1e6,'calls_per_unit':len(selected)/count,
          'groups':g,'kernels':k,'head_markers':len(heads)}
    result[test]=data
    for filename,items in [('groups.csv',g),('kernels.csv',k)]:
        with (path.parent/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    print('\n'+test, json.dumps({key:value for key,value in data.items() if key not in ['groups','kernels']}))
    for row in g[:14]:print(f"{row['total_ms_per_unit']:8.3f} ms {row['gpu_time_pct']:5.1f}% {row['calls_per_unit']:6.1f} calls {row['name']} scratch={row['max_scratch_bytes']}")
(root/'summary.json').write_text(json.dumps(result,indent=2))
