#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
exec >> training.log 2>&1
trap 'printf "%s\n" "$?" > training.exit' EXIT
until test -s corpus.exit; do sleep 20; done
test "$(cat corpus.exit)" = 0
test -s augmented/build_manifest.json && test -s image-qa.json
bash run_model.sh T4 "" &
t4_pid=$!
available_gpus() {
    nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits |
        awk -F ', ' '$1 <= 2 && $2 < 35 && $3 > 16000 {print $1}'
}
mapfile -t cards < <(available_gpus)
while test "${#cards[@]}" -eq 0; do
    sleep 30
    mapfile -t cards < <(available_gpus)
done
echo "Training T1 on GPU ${cards[0]}"
bash run_model.sh T1 "${cards[0]}" &
t1_pid=$!
status=0
if test "${#cards[@]}" -gt 1; then
    echo "Training the T3 image encoder on GPU ${cards[1]}"
    bash run_model.sh T3 "${cards[1]}" &
    t3_pid=$!
    wait "$t1_pid" || status=1
    wait "$t3_pid" || status=1
else
    wait "$t1_pid" || status=1
    bash run_model.sh T3 "${cards[0]}" || status=1
fi
wait "$t4_pid" || status=1
exit "$status"
