#!/usr/bin/env bash
PRED=D:/ML_Markowitz_data/preds/cn
LOG=$PRED/run_real.log
count() { find "$PRED" -name "*.parquet" | wc -l; }
stalls=0
while true; do
    before=$(count)
    python -X faulthandler -u run_real.py "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== chain exit code $rc $(date '+%F %T')" >> "$LOG"
    [ $rc -eq 0 ] && break
    if [ "$(count)" -gt "$before" ]; then stalls=0; else stalls=$((stalls + 1)); fi
    [ $stalls -ge 3 ] && { echo "=== giving up: three restarts without progress" >> "$LOG"; exit 1; }
done
