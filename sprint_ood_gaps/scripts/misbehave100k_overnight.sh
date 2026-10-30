#!/bin/bash
# 100k misbehaviour search, overnight finish (designs/misbehave_100k.md): wait for the screen retry, wait for Google's daily quota reset,
# full-grade each shard (Google direct, OpenRouter fallback for blocks). The pooled analysis is run by hand after the reader route check.
cd /home/cadcad/research/ood-gaps; L=runs/misbehave100k/overnight.log
while pgrep -f "screen_retry.sh" > /dev/null; do sleep 60; done
echo "screen retry finished $(date -u)" >> $L
now=$(date -u +%s); target=$(date -u -d "today 07:35" +%s); [ $now -gt $target ] && target=$(date -u -d "tomorrow 07:35" +%s)
sleep $(( target - now > 0 ? target - now : 0 ))
echo "grading start $(date -u)" >> $L
for k in 0 1 2 3; do
  ROUTE=direct .venv/bin/python scripts/misbehave100k_grade_shard.py runs/misbehave100k/shards/shard$k.jsonl runs/pod_sync/mb100k-shard$k/rollouts/rollouts.jsonl runs/misbehave100k/graded_shard$k 2> runs/misbehave100k/grade_shard$k.err | tail -20 >> $L
  [ -f runs/misbehave100k/graded_shard$k/shard_summary.json ] || { echo "shard $k did not finish; stopping $(date -u)" >> $L; exit 1; }
done
echo "grading done $(date -u); analysis waits for the codex-vs-OpenRouter reader check" >> $L
