import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("HGM_DATA", os.path.join(ROOT, "data", "raw"))
MEMORY = os.environ.get("HGM_MEMORY", "titans")
CKPT_BASE = os.path.join(ROOT, "checkpoints")
CKPT = CKPT_BASE if MEMORY == "titans" else os.path.join(CKPT_BASE, MEMORY)
THIRD = os.path.join(ROOT, "third_party")
ONLINETSF = os.path.join(THIRD, "OnlineTSF")
TCPD = os.path.join(THIRD, "TCPD")
OUT = os.environ.get("HGM_OUT", os.path.join(ROOT, "outputs"))
POLICY = os.path.join(OUT, "policies")
DIAG = os.path.join(OUT, "results")
FIGS = os.path.join(OUT, "figures")
TABLES = os.path.join(ROOT, "outputs", "tables")
LOGS = os.path.join(OUT, "logs")

for _d in (DATA, CKPT, POLICY, DIAG, FIGS, LOGS, TABLES):
    os.makedirs(_d, exist_ok=True)
