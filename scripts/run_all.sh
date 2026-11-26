#!/usr/bin/env bash
cd "$(dirname "$0")/.."; mkdir -p outputs/logs
PY="${PYTHON:-python}"
CMD="$PY main.py setup && $PY main.py main && $PY main.py generality && $PY main.py robustness && \
$PY main.py delayed && $PY main.py tcpd && $PY main.py posthoc && bash scripts/reproduce_paper.sh"
if command -v caffeinate >/dev/null; then nohup caffeinate -dims bash -c "$CMD" >> outputs/logs/run_all.log 2>&1 &
else nohup bash -c "$CMD" >> outputs/logs/run_all.log 2>&1 & fi
echo "started; follow with: tail -f outputs/logs/run_all.log"
