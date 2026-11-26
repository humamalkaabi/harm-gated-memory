from hgm import paths as HP
import re, subprocess, sys

W, Hc = 2000, 1180
o = []
FONT = "'Liberation Serif','Times New Roman',serif"
C = dict(org="#F6CBA9", orgS="#8B6A55", blu="#DCE6F5", bluS="#2F4F8F", yel="#FBE9B0", yelS="#9A7B1C",
         grn="#DDEFD9", grnS="#3E7A35", red="#F9E1DE", redS="#B03A2E", gry="#EEEEEE", gryS="#777", pur="#E9E0F3", purS="#5B3E8A")


def tx(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="72%">\1</tspan>', s)
    s = re.sub(r"\^\{([^}]*)\}", r'<tspan baseline-shift="super" font-size="72%">\1</tspan>', s)
    return s


def text(x, y, s, size=22, anchor="middle", weight="normal", style="normal", fill="#111"):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
             f'font-style="{style}" fill="{fill}">{tx(s)}</text>')


def rect(x, y, w, h, fill, stroke, sw=2, r=12, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def box(x, y, w, h, fill, stroke, lines, size=21, bold_first=True, sw=2, dash=None):
    rect(x, y, w, h, fill, stroke, sw, dash=dash)
    n = len(lines); lh = size * 1.3; y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, l in enumerate(lines):
        text(x + w / 2, y0 + i * lh, l, size if i else size + 1, weight="bold" if (bold_first and i == 0) else "normal")


def arrow(x1, y1, x2, y2, col="#333", sw=2.5, dash=None, mid=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    if mid:
        o.append(f'<path d="M{x1},{y1} {mid} {x2},{y2}" fill="none" stroke="{col}" stroke-width="{sw}"{d} marker-end="url(#a{col[1:]})"/>')
    else:
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}"{d} marker-end="url(#a{col[1:]})"/>')


def badge(x, y, ok, s, size=18):
    col = C["grnS"] if ok else C["redS"]; bg = C["grn"] if ok else C["red"]
    w = len(s) * size * 0.5 + 44
    rect(x - w / 2, y - 18, w, 34, bg, col, 1.8, r=17)
    text(x - w / 2 + 18, y + 6, "✓" if ok else "✗", size + 2, weight="bold", fill=col)
    text(x + 12, y + 6, s, size, fill=col)


def panel(x, y, w, h, label, title, fill, stroke):
    rect(x, y, w, h, fill, stroke, 2.2, r=16, dash="10 6")
    text(x + 18, y + 34, label, 24, anchor="start", weight="bold", fill=stroke)
    text(x + 70, y + 34, title, 22, anchor="start", weight="bold", fill=stroke)


cols = ["#333", C["bluS"], C["redS"], C["grnS"], C["purS"], C["gryS"], C["yelS"]]
o.append("<defs>" + "".join(
    f'<marker id="a{c[1:]}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{c}"/></marker>'
    for c in cols) + "</defs>")

panel(30, 60, 560, 820, "M1", "Host forecaster with test-time memory", "#FBFCFE", C["bluS"])
box(70, 115, 480, 70, C["gry"], C["gryS"], ["Input stream x_{≤t}", "lookback L, channel-independent"], 19)
box(70, 215, 480, 64, C["blu"], C["bluS"], ["RevIN + RLinear backbone", "Linear(x_{t}),  slow weights θ"], 19)
rect(70, 305, 480, 270, "#FFFFFF", C["bluS"], 2)
text(310, 338, "Test-time memory (per variate)", 21, weight="bold")
text(310, 372, "k_{t} = norm(W_{k} Enc(x_{t})),   l_{t} = ||M k_{t} − v_{t}||^{2}", 19)
text(310, 404, "S_{t} = η_{t} S_{t−1} − θ_{t} ∇l_{t},   M_{t} = (1 − α_{t}) M_{t−1} + S_{t}", 19)
text(310, 440, "surprise  s_{t} = ||∇l_{t}|| = 2||M k_{t} − v_{t}||", 19, fill=C["redS"], weight="bold")
text(310, 468, "(Proposition 1: associative reconstruction error)", 16, style="italic", fill=C["redS"])
box(95, 488, 430, 70, C["org"], C["orgS"], ["Forget-and-Recall (FR) controller", "error event → α boost · snapshot bank · delayed recall"], 17)
box(70, 600, 480, 60, C["blu"], C["bluS"], ["Forecast  ŷ_{t} = Linear(x_{t}) + MLP([M_{t}k_{t}, h_{t}])"], 18, bold_first=False)
box(70, 690, 480, 66, C["yel"], C["yelS"], ["Online adaptation policies", "delayed GD on θ · PROCEED · Online-GD"], 18)
text(310, 790, "independent, continuously maintained policies b, a ∈ 𝒜", 17, style="italic", fill=C["yelS"])
text(310, 820, "(no attention anywhere)", 17, style="italic", fill="#555")
arrow(310, 185, 310, 213); arrow(310, 279, 310, 303); arrow(310, 575, 310, 598); arrow(310, 660, 310, 688)

rect(30, 920, 1940, 70, "#F4F4F4", C["gryS"], 2, r=14, dash="6 5")
text(1000, 952, "Delayed feedback buffer  ℱ_{t}:  inputs up to t,  losses l_{o} of origins o ≤ t − H only", 22, weight="bold", fill="#333")
text(1000, 978, "no look-ahead · non-overlapping blocks of H origins for statistics", 18, style="italic", fill="#555")
arrow(310, 880, 310, 918, col=C["gryS"])
text(330, 905, "ŷ_{o} scored at o + H", 17, anchor="start", style="italic", fill="#555")

panel(630, 60, 520, 820, "M2", "Causal signals from ℱ_{t}", "#FFFDF8", C["yelS"])
sx, sw_ = 665, 450
box(sx, 115, sw_, 120, C["red"], C["redS"], ["Memory surprise  s_{t}", "gradient norm of the", "associative loss"], 19)
box(sx, 265, sw_, 120, C["gry"], C["gryS"], ["Input novelty  n_{t}", "mean / scale shift of the", "recent window"], 19)
box(sx, 415, sw_, 150, C["grn"], C["grnS"], ["Revealed-risk monitor  r_{t}",
                                              "r_{t} = log( mean l_{o} over O_{t}^{W}  /  l_{val} )",
                                              "last W fully revealed origins"], 19)
box(sx, 595, sw_, 175, C["pur"], C["purS"], ["Paired-utility estimator",
                                              "d_{o} = l_{o}(b) − l_{o}(a),  block means",
                                              "Bayesian local level → P(U_{t}(a) > 0 | ℱ_{t})",
                                              "vs. LCB rule, Hedge, static average"], 18)
arrow(550, 440, 663, 175, col=C["redS"], mid="Q600,300")
arrow(550, 150, 663, 325, col=C["gryS"], mid="Q610,240")
arrow(550, 640, 663, 680, col=C["purS"], mid="Q605,700")
for xx in (760, 890, 1020):
    pass
arrow(1130, 918, 1117, 500, col=C["grnS"], dash="7 5", mid="L1130,500")
arrow(1010, 918, 1010, 772, col=C["purS"], dash="7 5")

panel(1190, 60, 400, 820, "M3", "Online estimands", "#FAFFF9", C["grnS"])
ex, ew = 1220, 340
box(ex, 115, ew, 150, "#FFFFFF", C["gryS"], ["Change  D_{t}", "input law over [t, t+W)", "differs from [t−W, t)"], 19)
box(ex, 360, ew, 170, "#FFFFFF", C["redS"], ["Harm  R_{t}", "log of future risk of the", "deployed forecaster over", "its reference risk"], 19)
box(ex, 610, ew, 160, "#FFFFFF", C["purS"], ["Utility  U_{t}(a)", "future loss of reference b", "minus that of policy a"], 19)
arrow(1115, 160, 1218, 175, col=C["redS"]); badge(1390, 300, True, "s → D: TCPD AUROC 0.68", 16)
arrow(1115, 205, 1218, 420, col=C["redS"], dash="6 5")
arrow(1115, 325, 1218, 200, col=C["gryS"], dash="6 5")
arrow(1115, 490, 1218, 450, col=C["grnS"], sw=3.5)
arrow(1115, 682, 1218, 690, col=C["purS"], dash="6 5")
badge(1390, 553, True, "r → R: AUROC 0.76–0.85", 16)
badge(1390, 585, False, "s → R: ≈ 0.5 (chance)", 16)
badge(1390, 800, False, "U unpredictable: 0.46–0.54", 16)
text(1390, 848, "Prop. 2: D, R, U do not imply one another", 17, style="italic", fill="#333")
text(1390, 872, "Prop. 3: abrupt harm w/o precursor is indistinguishable", 15, style="italic", fill="#333")

panel(1630, 60, 340, 820, "M4", "", "#FDFBFF", C["purS"])
text(1800, 94, "Risk-aware uncertainty", 22, weight="bold", fill=C["purS"])
box(1655, 130, 290, 150, C["grn"], C["grnS"], ["Risk scale", "σ_{t}^{2} = EWMA of revealed", "squared errors (half-life W)", "≈ exp(r_{t}) · l_{val}"], 18)
box(1655, 315, 290, 180, C["pur"], C["purS"], ["Scaled conformal + ACI", "score  |y − ŷ| / σ_{o}", "interval  ŷ_{t} ± q(1−α_{t}) σ_{t}",
                                                 "α_{t}: ACI on delayed", "miscoverage"], 17)
box(1655, 530, 290, 120, C["pur"], C["purS"], ["Bayesian variant", "inverse-gamma variance with", "forgetting → Student-t"], 17)
box(1655, 690, 290, 130, "#FFFFFF", C["purS"], ["Output", "prediction intervals that", "keep coverage during harm"], 18)
arrow(1800, 280, 1800, 313, col=C["purS"]); arrow(1800, 495, 1800, 528, col=C["purS"]); arrow(1800, 650, 1800, 688, col=C["purS"])
arrow(1562, 445, 1653, 205, col=C["grnS"], sw=3.5, mid="Q1610,300")
text(1608, 1060, "", 10)

rect(30, 1020, 1940, 120, "#FFF8EC", C["orgS"], 2.2, r=16)
text(1000, 1062, "Decision principle: harm is anticipated (R predictable) but adaptation utility is not (U unpredictable);", 23, weight="bold", fill="#5A3E2B")
text(1000, 1100, "therefore act on anticipated harm through calibrated uncertainty (M4) rather than through more adaptation or memory writes.", 22, fill="#5A3E2B")
text(1000, 1128, "Evidence: ETTh1 (dev), ETTh2, ETTm1, Weather; utility and uncertainty confirmed on ETTh2 without changes.", 17, style="italic", fill="#7A5E4B")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hc}" viewBox="0 0 {W} {Hc}" font-family="{FONT}">'
       f'<rect width="{W}" height="{Hc}" fill="#FFFFFF"/>' + "".join(o) + "</svg>")
open(HP.FIGS + "/Fig1_framework.svg", "w").write(svg)
html = f'<html><body style="margin:0;background:#fff">{svg}</body></html>'
open(HP.FIGS + "/Fig1_framework.html", "w").write(html)
