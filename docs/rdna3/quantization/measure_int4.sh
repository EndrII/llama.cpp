#!/bin/bash
# usage: measure_int4.sh <tag> <model.gguf> [VAR=val ...]   -> KLD (24 chunks vs BF16), full PPL, speed; logs in work/m_<tag>_*.log
source /mnt/data/projects/ossicle-gguf/scripts/xt_env.sh
tag=$1; model=$2; shift 2
B=/mnt/data/projects/llama.cpp/build-rocm101/bin; W=/mnt/data/projects/ossicle-gguf/work
$XT "$@" $B/llama-perplexity -m $model -f $W/wiki.test.raw -c 512 -ngl 99 --chunks 24 --kl-divergence-base $W/kld_base.bin --kl-divergence > $W/m_${tag}_kld.log 2>&1
grep -E "Mean    KLD|Same top p|Mean    PPL\(Q\)" $W/m_${tag}_kld.log | tr -s ' ' | tr '\n' ';'; echo
if [ -z "$NOBENCH" ]; then
$XT "$@" $B/llama-bench -m $model -ngl 99 -p 512,2048 -n 128 -r 3 > $W/m_${tag}_bench.log 2>&1
grep -E "pp512|pp2048|tg128" $W/m_${tag}_bench.log | awk -F'|' '{print $6,$7,$8}' | tr -s ' '
fi
