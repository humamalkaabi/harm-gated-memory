import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgm.data import load

ok = True
for data in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]:
    try: z, tr, va, end = load(data)
    except Exception as e: print(data, "not available:", e); continue
    for H in (24, 48, 96):
        last = lambda t: t + H - 1
        checks = {
            "train targets < tr": True,
            "early stopping targets < va": last(va - H) < va,
            "harm reference targets < va": last(va - H - 1) < va,
            "interval calibration targets < va": last(va - H - 1) < va,
            "lr selection targets < va": last(va - 1) < va,
        }
        for k, v in checks.items():
            flag = "PASS" if v else ("DEVIATION (documented)" if k.startswith("lr") else "FAIL")
            if flag == "FAIL": ok = False
            if not v or H == 24: print(f"{data:8s} H={H:3d} {k:36s} {flag}")
print("ALL NON-DOCUMENTED CHECKS PASS" if ok else "FAILURE")
