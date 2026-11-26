#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" -c "import numpy, scipy, matplotlib" 2>/dev/null || { echo "Python with numpy, scipy and matplotlib needed (PYTHON=...)"; exit 1; }
REP=outputs/reproduce_report.txt; : > "$REP"
rm -f outputs/fix_report.txt outputs/e6_report.txt outputs/same_model.txt
run() { echo "===== $*" | tee -a "$REP"; "$PY" -m "$@" >> "$REP" 2>&1 || echo "FAILED: $*" | tee -a "$REP"; }
echo "python=$PY   $(date)" | tee -a "$REP"
run hgm.analysis.tables
run hgm.analysis.tables_results
run hgm.analysis.tab_accuracy
run hgm.analysis.review_tables outputs/results outputs/results_review
run hgm.analysis.supp_tables
run hgm.analysis.labels_text
run hgm.analysis.sens_summary
run hgm.analysis.fix_summary
run hgm.analysis.e6_summary
run hgm.analysis.same_model_summary
run hgm.analysis.fig_results
run hgm.analysis.fig_teaser
if [ -f outputs/policies/ETTh2_H24_s1.npz ]; then
  run hgm.analysis.figures
else
  echo "skipped Figures S1-S4: run  python main.py policies --data ETTh2 --seed 1  first" | tee -a "$REP"
fi
for f in outputs/fix_report.txt outputs/e6_report.txt outputs/same_model.txt outputs/review_summary.json; do
  [ -f "$f" ] && { echo "===== $f" >> "$REP"; cat "$f" >> "$REP"; }
done
echo "done -> outputs/tables/, outputs/figures/, $REP"
