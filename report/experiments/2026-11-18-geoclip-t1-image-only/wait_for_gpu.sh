#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
gpu=GPU-35cc307f-4b68-8336-2c05-d720e966140e
printf '%s Waiting for T2 to finish and GPU 1 to be available.\n' "$(date -u)"
while true; do
    if screen -ls | grep -F '.zhang7-geoclip-t2' >/dev/null; then
        sleep 30
        continue
    fi
    if memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) &&
       applications=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) &&
       [[ "$memory" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] &&
       (( memory < 100 )) && ! grep -Fx "$gpu" <<<"$applications" >/dev/null; then
        break
    fi
    sleep 30
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
printf '%s GPU 1 available, starting image-only run.\n' "$(date -u)"
exec bash launch.sh
