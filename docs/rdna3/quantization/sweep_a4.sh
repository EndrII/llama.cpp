#!/bin/bash
# usage: sweep_a4.sh <model.gguf> <out.tsv> "name:a4list" ...   (KLD vs BF16 + median pp512/pp2048/tg128)
cd /mnt/data/projects/ossicle-gguf
m=$1; out=$2; shift 2
for cfg in "$@"; do
  name=${cfg%%:*}; list=${cfg#*:}
  NOBENCH=1 scripts/measure_int4.sh sw_$name $m GGML_INT4=3 GGML_INT4_A4=$list GGML_INT4_CLIP=${CLIP:-0.85} > /dev/null
  kld=$(grep "Mean    KLD" work/m_sw_${name}_kld.log | awk '{print $3}'); top=$(grep "Same top p" work/m_sw_${name}_kld.log | awk '{print $4}')
  sp=$(scripts/bench_med.sh $m GGML_INT4=3 GGML_INT4_A4=$list GGML_INT4_CLIP=${CLIP:-0.85} | awk '{printf "%s ", $2}')
  echo -e "$name\t$list\t$kld\t$top\t$sp" | tee -a $out
done
