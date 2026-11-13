from pathlib import Path
import shutil,os
import pymupdf as fitz
r=Path('/home/svt/.config/Claude/scratch-workspaces/117fffdf-33f1-49e9-9021-541b4b295944/da0f735d-531e-47d2-8484-5ba02f97cc28/scratch-2026-10-30-635bae/report')
b=Path('/mnt/data/tuning27b-next/report-table-backup');b.mkdir(exist_ok=True)
tex=r/'qwen35_int4_summary.tex';pdf=r/'qwen35_int4_summary.pdf'
for p in [tex,pdf]:
 if not (b/p.name).exists():shutil.copy2(p,b/p.name)
s=tex.read_text();a=s.index('\\begin{table}[H]',s.index('\\label{sec:gpus}'));z=s.index('\\end{table}',a)+len('\\end{table}')
new=r'''\begin{table}[H]
\centering\fontsize{7.8}{9.3}\selectfont
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Qwen3.5-4B} & \multicolumn{2}{c}{Qwen3.5-9B} & \multicolumn{2}{c}{Qwen3.8-27B}\\
GPU & prefill & decode & prefill & decode & prefill & decode\\
\midrule
RTX 5090 & 16{,}036 & 313.6 & 11{,}721 & 220.5 & 3{,}881 & 79\\
RTX 4090 & 14{,}882 & 220.2 & 9{,}762 & 141.6 & 3{,}417 & 51\\
\textbf{RX 7900 XT (INT4 \tx{fast}, this work)} & \textbf{11{,}012} & \textbf{200.6} & \textbf{6{,}581} & \textbf{125.8} & \textbf{2{,}013} & \textbf{50}\\
RTX 5080 & 10{,}057 & 200.5 & 6{,}543 & 136.1 & -- & 50\\
RTX 5070 Ti & 8{,}889 & 187.6 & 5{,}872 & 126.0 & -- & --\\
RTX 5070 & 7{,}449 & 154.5 & 4{,}652 & 101.3 & -- & --\\
RTX 3090 & 7{,}366 & 179.1 & 4{,}632 & 122.1 & 1{,}507 & 43\\
RTX 5060 Ti 16\,GB & 4{,}776 & 107.6 & 2{,}955 & 70.0 & -- & --\\
\bottomrule
\end{tabular*}
\caption{Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B; local pp512/tg128, no MTP/speculation. 27B decode rounded to whole tokens/s; RTX~5080 27B: reported IQ4\_XS. -- = unavailable.}
\end{table}'''
tex.write_text(s[:a]+new+s[z:]);assert tex.read_text()[:a]==s[:a] and tex.read_text()[a+len(new):]==s[z:]
doc=fitz.open(pdf);pg=doc[1];rect=fitz.Rect(67,367.5,528,519)
pg.add_redact_annot(rect,fill=(1,1,1));pg.apply_redactions(images=0,graphics=2,text=0)
regular=doc.extract_font(15)[3];bold=doc.extract_font(13)[3]
texts=['Qwen3.5-4B','Qwen3.5-9B','Qwen3.8-27B','GPU','prefill','decode','RX 7900 XT (INT4 fast, this work)','RTX 5060 Ti 16 GB','reported IQ4_XS','no MTP/speculation']
for name,buf,fb in [('TblR',regular,'/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'),('TblB',bold,'/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf')]:
 font=fitz.Font(fontbuffer=buf)
 if all(font.has_glyph(ord(c)) for t in texts for c in t):pg.insert_font(fontname=name,fontbuffer=buf)
 else:pg.insert_font(fontname=name,fontfile=fb);font=fitz.Font(fontfile=fb)
 if name=='TblR':fr=font
 else:fbold=font

def text(t,x,y,size=7.8,bold=False,align='left'):
 f=fbold if bold else fr;w=f.text_length(t,fontsize=size)
 if align=='center':x-=w/2
 if align=='right':x-=w
 pg.insert_text((x,y),t,fontname='TblB' if bold else 'TblR',fontsize=size,color=(0,0,0))
xs=[68,208]+[208+319*i/6 for i in range(1,7)]
pg.draw_line((68,369.76),(527,369.76),width=.8)
for i,model in enumerate(['Qwen3.5-4B','Qwen3.5-9B','Qwen3.8-27B']):text(model,(xs[1+2*i]+xs[3+2*i])/2,380,8,bold=True,align='center')
text('GPU',68,391,7.6)
for c in range(6):text('prefill' if c%2==0 else 'decode',(xs[c+1]+xs[c+2])/2,391,7.4,align='center')
pg.draw_line((68,396.5),(527,396.5),width=.5)
rows=[['RTX 5090','16,036','313.6','11,721','220.5','3,881','79'],['RTX 4090','14,882','220.2','9,762','141.6','3,417','51'],['RX 7900 XT (INT4 fast, this work)','11,012','200.6','6,581','125.8','2,013','50'],['RTX 5080','10,057','200.5','6,543','136.1','--','50'],['RTX 5070 Ti','8,889','187.6','5,872','126.0','--','--'],['RTX 5070','7,449','154.5','4,652','101.3','--','--'],['RTX 3090','7,366','179.1','4,632','122.1','1,507','43'],['RTX 5060 Ti 16 GB','4,776','107.6','2,955','70.0','--','--']]
for i,row in enumerate(rows):
 y=409+i*11.3;isbold=i==2
 text(row[0],68,y,7.05 if isbold else 7.8,bold=isbold)
 for c,value in enumerate(row[1:]):text(value,xs[c+2]-2,y,7.8,bold=isbold,align='right')
pg.draw_line((68,493),(527,493),width=.8)
text('Table 1: Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B; local pp512/tg128, no MTP/speculation.',297.5,505,7.1,align='center')
text('27B decode rounded to whole tokens/s; RTX 5080 27B: reported IQ4_XS. -- = unavailable.',297.5,515,7.1,align='center')
tmp=pdf.with_suffix('.updated.pdf');doc.save(tmp,garbage=4,deflate=True);doc.close();os.replace(tmp,pdf)
updated=fitz.open(pdf);original=fitz.open(b/pdf.name);assert len(updated)==len(original)
for i in range(len(updated)):
 if i!=1:assert updated[i].get_text()==original[i].get_text(),i
# Pixel equality outside the table proves that the original page layout is preserved.
import numpy as np
scale=2
p0=original[1].get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
p1=updated[1].get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
a0=np.frombuffer(p0.samples,np.uint8).reshape(p0.height,p0.width,3);a1=np.frombuffer(p1.samples,np.uint8).reshape(p1.height,p1.width,3)
mask=np.ones(a0.shape[:2],bool);mask[int(rect.y0*scale)-2:int(rect.y1*scale)+2,int(rect.x0*scale)-2:int(rect.x1*scale)+2]=False
assert np.array_equal(a0[mask],a1[mask]),'Changes outside table'
updated[1].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save('/mnt/data/tuning27b-next/table_updated.png')
print('Updated only the comparison table in the report source and PDF; other pages and pixels outside the table are unchanged.')
