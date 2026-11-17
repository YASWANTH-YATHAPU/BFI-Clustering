"""Build the paper as an IEEE-style two-column PDF without a LaTeX installation.

    python paper/build_pdf.py          (run from the project root)  ->  paper/BFI_paper_IEEE.pdf

Same content as paper/main.tex.  Layout: US-letter, Times New Roman 10 pt, two columns, IEEE
section / table / figure conventions.  Equations are rendered with matplotlib mathtext; the result
tables are computed from results/exp1_raw.csv and results/exp1_summary.csv (nothing typed by hand).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table,
                                TableStyle, KeepTogether, NextPageTemplate, FrameBreak)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIG = os.path.join(PAPER, "figures")
EQD = os.path.join(PAPER, "_eq")
os.makedirs(EQD, exist_ok=True)
OUT = os.path.join(PAPER, "BFI_paper_IEEE.pdf")

# ------------------------------------------------------------------ fonts
FD = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("TNR", FD + "times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", FD + "timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", FD + "timesi.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", FD + "timesbi.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

# ------------------------------------------------------------------ geometry (IEEE journal)
PW, PH = letter
LM = RM = 45.0
TOP, BOT = 54.0, 54.0
GAP = 18.0
COLW = (PW - LM - RM - GAP) / 2
FULLW = PW - LM - RM
YTOP = PH - TOP
HEADER = "IEEE TRANSACTIONS ON FUZZY SYSTEMS (MANUSCRIPT DRAFT)"

# ------------------------------------------------------------------ styles
body = ParagraphStyle("body", fontName="TNR", fontSize=10, leading=11.6, alignment=TA_JUSTIFY, firstLineIndent=10)
body0 = ParagraphStyle("body0", parent=body, firstLineIndent=0)
item = ParagraphStyle("item", parent=body, firstLineIndent=0, leftIndent=14, bulletIndent=2, spaceAfter=1,
                      bulletFontName="TNR", bulletFontSize=10)
item2 = ParagraphStyle("item2", parent=item, leftIndent=26, bulletIndent=16)
title_st = ParagraphStyle("title", fontName="TNR", fontSize=23, leading=27, alignment=TA_CENTER)
auth_st = ParagraphStyle("auth", fontName="TNR", fontSize=11, leading=13, alignment=TA_CENTER, spaceBefore=8)
abs_st = ParagraphStyle("abs", fontName="TNR-B", fontSize=9, leading=10.4, alignment=TA_JUSTIFY, firstLineIndent=10)
h1 = ParagraphStyle("h1", fontName="TNR", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=8, spaceAfter=4)
h2 = ParagraphStyle("h2", fontName="TNR-I", fontSize=10, leading=12, alignment=TA_LEFT, spaceBefore=5, spaceAfter=2)
cap_tab = ParagraphStyle("captab", fontName="TNR", fontSize=8, leading=9.5, alignment=TA_CENTER)
cap_fig = ParagraphStyle("capfig", fontName="TNR", fontSize=8, leading=9.5, alignment=TA_JUSTIFY, spaceBefore=3)
cell = ParagraphStyle("cell", fontName="TNR", fontSize=7, leading=8.2, alignment=TA_CENTER)
cell_l = ParagraphStyle("celll", parent=cell, alignment=TA_LEFT)
note_st = ParagraphStyle("note", fontName="TNR", fontSize=7, leading=8.4, alignment=TA_JUSTIFY)
alg_st = ParagraphStyle("alg", fontName="TNR", fontSize=8.5, leading=10, alignment=TA_LEFT)
ref_st = ParagraphStyle("ref", fontName="TNR", fontSize=8, leading=9.3, alignment=TA_JUSTIFY, leftIndent=18,
                        firstLineIndent=-18, spaceAfter=1.5)
foot_st = ParagraphStyle("foot", fontName="TNR", fontSize=8, leading=9.3, alignment=TA_JUSTIFY, firstLineIndent=8)

# ------------------------------------------------------------------ inline math tokens
K1, K2 = "<i>K</i><sub>1</sub>", "<i>K</i><sub>2</sub>"
N = "<i>N</i>"
kmin, kmax = "<i>k</i><sub>min</sub>", "<i>k</i><sub>max</sub>"
th, de, ze, ga = "<i>θ</i>", "<i>δ</i>", "<i>ζ</i>", "<i>γ</i>"
Mt, St = "<i>M</i><sub><i>t</i></sub>", "<i>S</i><sub><i>t</i></sub>"
DAG = "S1+S2<super>†</super>"


def sc(text, big=10, small=8):
    """IEEE small-caps heading."""
    out = []
    for w in text.split(" "):
        if w and w[0].isalpha():
            out.append(f'<font size="{big}">{w[0].upper()}</font><font size="{small}">{w[1:].upper()}</font>')
        else:
            out.append(f'<font size="{big}">{w}</font>')
    return " ".join(out)


# ------------------------------------------------------------------ citations (numbered in order of first use)
REFS = {
    "zhang2025": "Z.-w. Zhang, Z.-g. Liu, L.-b. Ning, H.-p. Tian, and B.-l. Wang, “Belief-based fuzzy and imprecise clustering for arbitrary data distributions,” <i>IEEE Trans. Fuzzy Syst.</i>, vol. 33, no. 8, pp. 2755–2767, Aug. 2025.",
    "zhang2025survey": "Z. Zhang, Y. Zhang, H. Tian, A. Martin, Z. Liu, and W. Ding, “A survey of evidential clustering: Definitions, methods, and applications,” <i>Inf. Fusion</i>, vol. 115, 2025, Art. no. 102736.",
    "macqueen1967": "J. MacQueen, “Some methods for classification and analysis of multivariate observations,” in <i>Proc. 5th Berkeley Symp. Math. Statist. Probab.</i>, vol. 1, 1967, pp. 281–297.",
    "bezdek1981": "J. C. Bezdek, <i>Pattern Recognition With Fuzzy Objective Function Algorithms</i>. New York, NY, USA: Plenum, 1981.",
    "ester1996": "M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based algorithm for discovering clusters in large spatial databases with noise,” in <i>Proc. 2nd Int. Conf. Knowl. Discovery Data Mining</i>, 1996, pp. 226–231.",
    "comaniciu2002": "D. Comaniciu and P. Meer, “Mean shift: A robust approach toward feature space analysis,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 24, no. 5, pp. 603–619, May 2002.",
    "rodriguez2014": "A. Rodriguez and A. Laio, “Clustering by fast search and find of density peaks,” <i>Science</i>, vol. 344, no. 6191, pp. 1492–1496, 2014.",
    "du2016": "M. Du, S. Ding, and H. Jia, “Study on density peaks clustering based on k-nearest neighbors and principal component analysis,” <i>Knowl.-Based Syst.</i>, vol. 99, pp. 135–145, 2016.",
    "xie2016": "J. Xie, H. Gao, W. Xie, X. Liu, and P. W. Grant, “Robust clustering by detecting density peaks and assigning points based on fuzzy weighted K-nearest neighbors,” <i>Inf. Sci.</i>, vol. 354, pp. 19–40, 2016.",
    "guan2024": "J. Guan, S. Li, J. Zhu, X. He, and J. Chen, “Fast main density peak clustering within relevant regions via a robust decision graph,” <i>Pattern Recognit.</i>, vol. 152, 2024, Art. no. 110458.",
    "yu2026": "S. Yu, J. Engels, Y. Huang, and J. Shun, “PECANN: Parallel efficient clustering with graph-based approximate nearest neighbor search,” in <i>Proc. Conf. Appl. Comput. Discrete Algorithms</i>, 2025, pp. 1–17.",
    "masson2008": "M.-H. Masson and T. Denœux, “ECM: An evidential version of the fuzzy c-means algorithm,” <i>Pattern Recognit.</i>, vol. 41, no. 4, pp. 1384–1397, 2008.",
    "su2019": "Z.-g. Su and T. Denœux, “BPEC: Belief-peaks evidential clustering,” <i>IEEE Trans. Fuzzy Syst.</i>, vol. 27, no. 1, pp. 111–123, Jan. 2019.",
    "dempster1967": "A. P. Dempster, “Upper and lower probabilities induced by a multivalued mapping,” <i>Ann. Math. Statist.</i>, vol. 38, no. 2, pp. 325–339, 1967.",
    "shafer1976": "G. Shafer, <i>A Mathematical Theory of Evidence</i>. Princeton, NJ, USA: Princeton Univ. Press, 1976.",
    "smets1990": "P. Smets, “The combination of evidence in the transferable belief model,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 12, no. 5, pp. 447–458, May 1990.",
    "denoeux2004": "T. Denœux and M.-H. Masson, “EVCLUS: Evidential clustering of proximity data,” <i>IEEE Trans. Syst., Man, Cybern. B, Cybern.</i>, vol. 34, no. 1, pp. 95–109, Feb. 2004.",
    "liu2015": "Z.-g. Liu, Q. Pan, J. Dezert, and G. Mercier, “Credal C-means clustering method based on belief functions,” <i>Knowl.-Based Syst.</i>, vol. 74, pp. 119–132, 2015.",
    "qiu2023": "T. Qiu and Y.-J. Li, “Fast LDP-MST: An efficient density-peak-based clustering method for large-size datasets,” <i>IEEE Trans. Knowl. Data Eng.</i>, vol. 35, no. 5, pp. 4767–4780, May 2023.",
    "bentley1975": "J. L. Bentley, “Multidimensional binary search trees used for associative searching,” <i>Commun. ACM</i>, vol. 18, no. 9, pp. 509–517, 1975.",
    "friedman1977": "J. H. Friedman, J. L. Bentley, and R. A. Finkel, “An algorithm for finding best matches in logarithmic expected time,” <i>ACM Trans. Math. Softw.</i>, vol. 3, no. 3, pp. 209–226, 1977.",
    "weber1998": "R. Weber, H.-J. Schek, and S. Blott, “A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces,” in <i>Proc. 24th Int. Conf. Very Large Data Bases</i>, 1998, pp. 194–205.",
    "dong2011": "W. Dong, M. Charikar, and K. Li, “Efficient k-nearest neighbor graph construction for generic similarity measures,” in <i>Proc. 20th Int. Conf. World Wide Web</i>, 2011, pp. 577–586.",
    "malkov2020": "Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 42, no. 4, pp. 824–836, Apr. 2020.",
    "pedregosa2011": "F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine learning in Python,” <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.",
    "hubert1985": "L. Hubert and P. Arabie, “Comparing partitions,” <i>J. Classification</i>, vol. 2, no. 1, pp. 193–218, 1985.",
    "strehl2002": "A. Strehl and J. Ghosh, “Cluster ensembles—A knowledge reuse framework for combining multiple partitions,” <i>J. Mach. Learn. Res.</i>, vol. 3, pp. 583–617, 2002.",
    "kuhn1955": "H. W. Kuhn, “The Hungarian method for the assignment problem,” <i>Naval Res. Logistics Quart.</i>, vol. 2, no. 1–2, pp. 83–97, 1955.",
    "harris2020": "C. R. Harris <i>et al.</i>, “Array programming with NumPy,” <i>Nature</i>, vol. 585, pp. 357–362, 2020.",
    "virtanen2020": "P. Virtanen <i>et al.</i>, “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” <i>Nature Methods</i>, vol. 17, pp. 261–272, 2020.",
}
ORDER = []


def c(keys):
    out = []
    for k in keys.split(","):
        k = k.strip()
        if k not in REFS:
            raise KeyError(k)
        if k not in ORDER:
            ORDER.append(k)
        out.append(f"[{ORDER.index(k) + 1}]")
    return ", ".join(out)


# ------------------------------------------------------------------ equations via mathtext
plt.rcParams["mathtext.fontset"] = "stix"
_eq_n = [0]


def EQ(tex, size=10.5):
    _eq_n[0] += 1
    n = _eq_n[0]
    path = os.path.join(EQD, f"eq{n}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    w, h = PILImage.open(path).size
    wpt, hpt = w / 300 * 72, h / 300 * 72
    if wpt > COLW - 30:
        hpt *= (COLW - 30) / wpt
        wpt = COLW - 30
    t = Table([[Image(path, width=wpt, height=hpt), Paragraph(f"({n})", body0)]],
              colWidths=[COLW - 26, 26])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


# ------------------------------------------------------------------ helpers
story = []
P = lambda t, st=body: story.append(Paragraph(t, st))
H1 = lambda t: story.append(Paragraph(t, h1))
H2 = lambda t: story.append(Paragraph(t, h2))


def ITEMS(lst, numbered=True, st=item, labels=None):
    for i, t in enumerate(lst):
        b = labels[i] if labels else (f"{i + 1})" if numbered else "•")
        story.append(Paragraph(t, st, bulletText=b))


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def booktabs(data, widths, header_rows=1, extra=None, fs=7):
    t = Table(data, colWidths=widths, repeatRows=0)
    st = [("FONTNAME", (0, 0), (-1, -1), "TNR"), ("FONTSIZE", (0, 0), (-1, -1), fs),
          ("LEADING", (0, 0), (-1, -1), fs + 1.2), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.1),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.1), ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
          ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
          ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black)]
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def table_block(num, caption, tbl, note=None, width=COLW):
    items = [Paragraph(f"TABLE {ROMAN[num - 1]}", cap_tab), Paragraph(sc(caption, 8, 6.5), cap_tab), Spacer(1, 3), tbl]
    if note:
        items += [Spacer(1, 2), Paragraph(note, note_st)]
    items.append(Spacer(1, 8))
    return items


from reportlab.platypus.flowables import Flowable


class FitFigure(Flowable):
    """Image + caption that shrinks to the space left in the column (down to min_h), otherwise moves on."""

    def __init__(self, path, caption, max_h=None, min_h=300):
        super().__init__()
        self.path, self.caption, self.max_h, self.min_h = path, caption, max_h, min_h
        self.iw, self.ih = PILImage.open(path).size

    def wrap(self, aw, ah):
        self.aw = aw
        self.cap = Paragraph(self.caption, cap_fig)
        ch = self.cap.wrap(aw, ah)[1]
        H = aw * self.ih / self.iw
        if self.max_h:
            H = min(H, self.max_h)
        avail = ah - ch - 14
        if H > avail and avail >= min(self.min_h, H):
            H = avail
        self.H, self.W, self.ch = H, H * self.iw / self.ih, ch
        return aw, H + ch + 14

    def split(self, aw, ah):
        return [FrameBreak(), FitFigure(self.path, self.caption, self.max_h, self.min_h)]

    def draw(self):
        self.canv.drawImage(self.path, (self.aw - self.W) / 2, self.ch + 8, self.W, self.H, mask="auto")
        self.cap.drawOn(self.canv, 0, 2)


def figure(num, fname, caption, width=COLW, max_h=None):
    story.append(FitFigure(os.path.join(FIG, fname), f"Fig. {num}. {caption}", max_h=max_h))


def algorithm(num, title, inp, par, lines):
    rows = [[Paragraph(f"<b>Algorithm {num}:</b> {title}", alg_st)],
            [Paragraph(f"<b>Input:</b> {inp}", alg_st)],
            [Paragraph(f"<b>Parameters:</b> {par}", alg_st)],
            [Paragraph("<b>Procedure:</b>", alg_st)]]
    for i, (ind, txt) in enumerate(lines):
        st = ParagraphStyle(f"al{num}{i}", parent=alg_st, leftIndent=12 + 10 * ind, firstLineIndent=-12)
        rows.append([Paragraph(f'<font size="7">{i + 1}</font>&nbsp;&nbsp;{txt}', st)])
    t = Table(rows, colWidths=[COLW])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
    story.append(KeepTogether([Spacer(1, 4), t, Spacer(1, 8)]))


# ------------------------------------------------------------------ result data
raw = pd.read_csv(os.path.join(ROOT, "results", "exp1_raw.csv"))
summ = pd.read_csv(os.path.join(ROOT, "results", "exp1_summary.csv")).set_index(["dataset", "method", "theta"])
METRICS = ["Re", "Ri", "P", "R", "RI", "EP", "ER", "ERI", "ARI", "AMI", "NMI"]
HEADM = ["<i>R<sub>e</sub></i>", "<i>R<sub>i</sub></i>", "<i>P</i>", "<i>R</i>", "<i>RI</i>", "<i>EP</i>",
         "<i>ER</i>", "<i>ERI</i>", "ARI", "AMI", "NMI"]
ROWS = [("original", "exact", "Original BFI"), ("control", "exact", "Control"), ("S1", "exact", "S1"),
        ("S2", "exact", "S2"), ("S1+S2", "exact", "S1+S2"), ("S1+S2", "sampled", DAG)]
NAMES = {"moons": "moons", "circles": "circles", "anisotropic": "aniso.", "blobs_separated": "blobs_sep.",
         "blobs_overlap": "blobs_ovl.", "varied_density": "varied", "blobs_hd8": "hd8", "blobs_hd32": "hd32",
         "iris": "iris", "wine": "wine", "breast_cancer": "breast", "digits6_pca10": "digits6",
         "digits10_pca16": "digits10"}
SYN = ["moons", "circles", "anisotropic", "blobs_separated", "blobs_overlap", "varied_density", "blobs_hd8", "blobs_hd32"]
REAL = ["iris", "wine", "breast_cancer", "digits6_pca10", "digits10_pca16"]
means = raw.groupby(["dataset", "method", "theta"])[METRICS].mean()


def metric_table(datasets):
    data = [[Paragraph("<b>Datasets</b>", cell), Paragraph("<b>Methods</b>", cell)] +
            [Paragraph(h, cell) for h in HEADM]]
    extra = []
    for ds in datasets:
        blk = np.round(np.array([[means.loc[(ds, m, t), k] for k in METRICS] for m, t, _ in ROWS]), 4)
        best = np.where(np.array([k == "Re" for k in METRICS]), blk.min(0), blk.max(0))
        r0 = len(data)
        for ri, (m, t, lab) in enumerate(ROWS):
            row = [NAMES[ds] if ri == 0 else "", Paragraph(lab, cell_l)]
            for ci, k in enumerate(METRICS):
                row.append(f"{blk[ri, ci]:.4f}")
                if k != "Ri" and blk[ri, ci] == best[ci]:
                    extra.append(("FONTNAME", (ci + 2, len(data)), (ci + 2, len(data)), "TNR-B"))
            data.append(row)
        extra += [("SPAN", (0, r0), (0, r0 + len(ROWS) - 1)),
                  ("LINEBELOW", (0, r0 + len(ROWS) - 1), (-1, r0 + len(ROWS) - 1), 0.3, colors.black)]
    widths = [44, 56] + [(FULLW - 100) / 11] * 11
    return booktabs(data, widths, extra=extra)


NOTE_WIDE = ("Mean over seeds (5 for synthetic data; real data are deterministic). All variants use exact "
             f"{th} except {DAG} ({th} sampled from 1000 objects). Bold: best value per dataset (lowest for "
             "<i>R<sub>e</sub></i>); <i>R<sub>i</sub></i> is not ranked.")


def wide_block(num, caption, datasets):
    items = table_block(num, caption, metric_table(datasets), NOTE_WIDE, FULLW)
    h = sum(f.wrap(FULLW, PH)[1] for f in items)
    return items, h


WIDE = {"syn": wide_block(1, "Clustering results of the original BFI and the accelerated variants on the eight synthetic datasets", SYN),
        "real": wide_block(2, "Clustering results of the original BFI and the accelerated variants on the five real-world datasets", REAL)}

# ------------------------------------------------------------------ page templates
AUTH_FOOT = ("Author Name 1, Author Name 2, and Author Name 3 are with the Department of …, … University, City, "
             "Country (e-mail: …). Replace these placeholders with the full names and affiliations of all team members.")
TITLE_H = 112
FOOT_H = 44


def header(canv, doc):
    canv.saveState()
    canv.setFont("TNR", 8)
    canv.drawString(LM, PH - 34, HEADER)
    canv.drawRightString(PW - RM, PH - 34, str(doc.page))
    canv.restoreState()


def first_page(canv, doc):
    header(canv, doc)
    canv.saveState()
    canv.setLineWidth(0.4)
    canv.line(LM, BOT + FOOT_H - 2, LM + 60, BOT + FOOT_H - 2)
    p = Paragraph(AUTH_FOOT, foot_st)
    w, h = p.wrap(COLW, FOOT_H)
    p.drawOn(canv, LM, BOT + FOOT_H - 6 - h)
    canv.restoreState()


def wide_page(key):
    def fn(canv, doc):
        header(canv, doc)
        items, h = WIDE[key]
        y = YTOP
        for f in items:
            w, fh = f.wrap(FULLW, PH)
            f.drawOn(canv, LM, y - fh)
            y -= fh
    return fn


def cols(y0, h, foot=0):
    return [Frame(LM, y0 + foot, COLW, h - foot, 0, 0, 0, 0, id="c1"),
            Frame(LM + COLW + GAP, y0, COLW, h, 0, 0, 0, 0, id="c2")]


first = PageTemplate("first", [Frame(LM, YTOP - TITLE_H, FULLW, TITLE_H, 0, 0, 0, 0, id="title")] +
                     cols(BOT, YTOP - TITLE_H - BOT - 6, FOOT_H), onPage=first_page, autoNextPageTemplate="later")
later = PageTemplate("later", cols(BOT, YTOP - BOT), onPage=header)
tpl = [first, later]
for key, nxt in (("syn", "real"), ("real", "later")):
    h = WIDE[key][1]
    tpl.append(PageTemplate(key, cols(BOT, YTOP - h - BOT - 4), onPage=wide_page(key), autoNextPageTemplate=nxt))
class Doc(BaseDocTemplate):
    """reportlab keeps _nextPageTemplateIndex after an automatic switch, which repeats the template once; clear it."""

    def _setPageTemplate(self):
        if hasattr(self, "_nextPageTemplateIndex"):
            self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
            del self._nextPageTemplateIndex
        elif self.pageTemplate.autoNextPageTemplate:
            self.handle_nextPageTemplate(self.pageTemplate.autoNextPageTemplate)
            self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
            del self._nextPageTemplateIndex


doc = Doc(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM, topMargin=TOP, bottomMargin=BOT,
                      title="An Efficient and Evidence-Preserving Framework for Belief-Based Fuzzy and Imprecise Clustering",
                      author="Author Names")
doc.addPageTemplates(tpl)

# ================================================================== CONTENT
story.append(Paragraph("An Efficient and Evidence-Preserving Framework for Belief-Based Fuzzy and Imprecise Clustering", title_st))
story.append(Paragraph("Author Name 1, Author Name 2, and Author Name 3", auth_st))
story.append(FrameBreak())

P(f"<i>Abstract</i>—Belief-based fuzzy and imprecise clustering (BFI) detects clusters of arbitrary shapes through "
  "belief-based density peaks and evidential convergence, and assigns ambiguous objects to meta-clusters. However, its "
  f"reference formulation computes and sorts the full {N}×{N} distance matrix, i.e., <i>O</i>({N}<super>2</super> log {N}) "
  f"time and <i>O</i>({N}<super>2</super>) memory, and its authors identify the long execution time as its main "
  "limitation. To address this issue, this article investigates an efficient and evidence-preserving BFI that changes "
  "only how neighbours are retrieved. First, an adaptive candidate neighbour retrieval gives every object a short, sorted, "
  "exact candidate list from a k-d tree. Sufficiency certificates prove when such a list answers the two search queries "
  "of BFI exactly, and only the lists that fail the certificate are enlarged. Second, an exact distance-based refinement "
  "computes the belief values from exact neighbour distances streamed in bounded memory and resolves the few uncertified "
  "queries by exact scans restricted to the relevant subset. With an exact distance factor, the proposed method reproduces "
  "the output of the original BFI exactly, including the meta-clusters (label agreement 1.000 ± 0.000 on 13 datasets × 5 "
  f"seeds). Peak memory falls from 366 MB to 26 MB at {N} = 4000, and 50 000 objects are clustered with 38 MB, where the "
  "distance matrix alone would need about 20 GB. The method is up to 2.3× faster at the largest size the original could "
  "run. The cases where the acceleration does not help are also reported.", abs_st)
story.append(Spacer(1, 4))
P("<i>Index Terms</i>—Clustering, density peaks, evidence theory, imprecision, nearest-neighbour search, scalability.", abs_st)
story.append(Spacer(1, 4))

# ---------------- I. Introduction
H1(sc("I. Introduction"))
P("REAL-WORLD datasets often contain complex structures characterized by irregular cluster shapes, varying densities, "
  "overlapping regions, and noisy observations. Such characteristics are commonly encountered in image segmentation, "
  "pattern recognition, medical image analysis, and other data-driven applications. For instance, in image segmentation, "
  "neighbouring pixels may exhibit similar features while pixels near object boundaries can share characteristics of "
  "multiple regions, making precise assignment difficult. Similarly, in multidimensional datasets, clusters may differ "
  "significantly in size and density, creating challenges for conventional clustering techniques. These characteristics "
  "make it important to develop clustering approaches that can identify complex data distributions while representing "
  "uncertainty and ambiguity in cluster membership.", body0)
P("To address these challenges, several unsupervised clustering approaches have been developed, including partition-based, "
  "fuzzy-based, and density-based techniques. Partition-based methods group observations according to their similarity to "
  "predefined cluster representatives, while fuzzy clustering assigns membership degrees to multiple clusters to represent "
  "gradual membership. Density-based approaches identify clusters through the distribution and concentration of "
  "neighbouring observations and can discover clusters with irregular shapes. More recently, evidential clustering methods "
  "have incorporated evidence theory to represent uncertainty and imprecision through belief assignments over individual "
  f"clusters and combinations of clusters {c('zhang2025survey')}.")
P(f"Among the existing approaches, K-means and fuzzy C-means are widely studied partition-based and fuzzy clustering methods, "
  f"respectively {c('macqueen1967,bezdek1981')}. DBSCAN identifies dense regions and separates noise {c('ester1996')}, "
  f"whereas Mean Shift detects clusters through density estimation {c('comaniciu2002')}. Density Peaks Clustering (DPC), "
  "introduced by Rodriguez and Laio, identifies potential cluster centres using local density and distance from "
  f"higher-density observations {c('rodriguez2014')}. Several extensions have subsequently investigated improvements in "
  f"scalability, centre detection, and data allocation {c('du2016,xie2016,guan2024,yu2026')}. Evidential clustering "
  f"methods further incorporate belief functions to represent ambiguous membership {c('masson2008,su2019')}. In 2025, "
  "Zhang <i>et al.</i> proposed Belief-Based Fuzzy and Imprecise Clustering (BFI), combining belief-based density-peak "
  "detection and evidential convergence to identify arbitrary-shaped clusters and represent imprecise observations "
  f"through meta-clusters {c('zhang2025')}.")
P("Despite these developments, computational efficiency remains an important consideration when applying density-based "
  "and evidential clustering methods to larger datasets. The original BFI framework requires neighbour information, "
  "distance calculations, and evidence-based assignment operations during its clustering process. Its authors acknowledge "
  "that the method requires a relatively long execution time, limiting its applicability in some cases, and identify "
  f"acceleration as future work {c('zhang2025')}. Moreover, acceleration must be evaluated not only in terms of execution "
  "time but also in terms of its impact on belief assignments, meta-cluster identification, and overall clustering "
  "quality. Without loss of generality, the question addressed in this article is as follows.")
P("<i>How can BFI be accelerated without changing its belief assignments and meta-clusters?</i>")
P("To this end, we propose an efficient and evidence-preserving extension of BFI based on adaptive neighbour search. The "
  "main contributions are summarized in the following three aspects.")
ITEMS([
    "We propose an <i>adaptive candidate neighbour retrieval</i> for BFI. Every object receives a short exact candidate "
    f"list, and <i>sufficiency certificates</i> prove when this list answers the {de} query and the evidential-convergence "
    "query exactly. Only lists that fail the certificate are enlarged, so the candidate size follows the local data structure.",
    "We design an <i>exact distance-based refinement</i>. It computes the belief values from exact neighbour distances "
    "streamed in bounded memory and resolves uncertified queries by exact scans restricted to the higher-belief or "
    "allocated subset. Together with the first contribution, it provably reproduces the belief assignments and "
    "meta-clusters of the original BFI without forming the distance matrix.",
    "We evaluate the method against the original BFI on 13 synthetic and real datasets with 5 seeds and on scalability "
    "data with up to 50 000 objects. The evaluation covers clustering quality, evidential metrics, runtime, memory and "
    "neighbour-search accuracy. It is complemented by a factorial ablation, a parameter-sensitivity study, a negative "
    "result and an explicit list of limitations."])
P("The rest of this article is organized as follows. Section II offers a concise overview of evidence theory, BFI and "
  "related works. Section III introduces the proposed method. Section IV is dedicated to assessing its efficiency and "
  "evidence preservation. In Section V, we discuss the complexity and limitations. Finally, Section VI concludes this article.")

# ---------------- II. Preliminaries
H1(sc("II. Preliminaries"))
P("In this section, we provide a brief overview of evidence theory, of the BFI method, and of related works on evidential "
  "clustering and efficient density-peaks clustering.", body0)
H2("A. Basics of Evidence Theory")
P(f"Evidence theory {c('dempster1967,shafer1976')} extends the frame Ω = {{<i>ω</i><sub>1</sub>, …, <i>ω<sub>c</sub></i>}} "
  "to the power set 2<super>Ω</super>. A basic belief assignment (BBA) <i>m</i>: 2<super>Ω</super> → [0, 1] satisfies", body0)
story.append(EQ(r"\sum_{A \in 2^{\Omega}} m(A) = 1."))
P("A subset <i>A</i> with <i>m</i>(<i>A</i>) &gt; 0 is a focal set. In clustering, {<i>ω<sub>k</sub></i>} is a singleton "
  "(precise) cluster, a subset with several clusters is a meta-cluster expressing imprecision, and ∅ is the noise "
  "cluster. Two BBAs are combined by Dempster's rule", body0)
story.append(EQ(r"[m_1 \oplus m_2](A) = \frac{\sum_{B \cap C = A} m_1(B)\, m_2(C)}{1 - \sum_{B \cap C = \emptyset} m_1(B)\, m_2(C)}."))
P(f"A crisp decision is taken from the pignistic probability {c('smets1990')}", body0)
story.append(EQ(r"\mathrm{BetP}(\omega_k) = \sum_{A \subseteq \Omega,\ \omega_k \in A} \frac{m(A)}{|A|}."))

H2("B. Belief-Based Fuzzy and Imprecise Clustering")
P(f"Consider a dataset <i>X</i> = {{<i>x</i><sub>1</sub>, …, <i>x<sub>N</sub></i>}} with <i>d<sub>ij</sub></i> = ‖<i>x<sub>i</sub></i> − "
  f"<i>x<sub>j</sub></i>‖, and let <i>d</i><sub><i>i</i>,(<i>k</i>)</sub> be the <i>k</i>-th smallest distance from "
  f"<i>x<sub>i</sub></i> to another object. BFI {c('zhang2025')} consists of two steps.", body0)
P(f"<i>1) Belief-based density peaks:</i> Each of the {K1} nearest neighbours <i>x<sub>k</sub></i> of <i>x<sub>i</sub></i> "
  "provides a mass function on {<i>C</i>, <i>U</i>} (centre / unknown) with <i>m<sub>ik</sub></i>(<i>C</i>) = "
  f"e<super>−<i>θ<sub>i</sub>d<sub>ik</sub></i></super>/{K1}, where the distance factor is")
story.append(EQ(r"\theta_i = 1 / \mathrm{quantile}\left(\{d_{ij} : j \neq i\},\ q\right),\quad q = 0.9."))
P("Combining the mass functions by (2) gives the belief that <i>x<sub>i</sub></i> is a centre:", body0)
story.append(EQ(r"\mathrm{Bel}_i(C) = 1 - \prod_{k=1}^{K_1} \left(1 - \frac{1}{K_1}\, e^{-\theta_i d_{i,(k)}}\right)."))
P("The distance to the nearest object of higher belief is", body0)
story.append(EQ(r"\delta_i = \min_{j:\ \mathrm{Bel}_j > \mathrm{Bel}_i} d_{ij} \quad (\delta_i = \max_j d_{ij}\ \mathrm{for\ the\ top\ object}),"))
P(f"and the <i>c</i> objects with the largest {ga}<sub><i>i</i></sub> in the decision graph are selected as cluster centres.", body0)
P(f"<i>2) Evidential convergence:</i> The remaining objects are allocated in decreasing order of Bel. For object "
  f"<i>x<sub>t</sub></i>, the {K2} nearest <i>already-allocated</i> objects {St} each provide "
  f"<i>m</i>({{<i>ω<sub>φ</sub></i>}}) = e<super>−<i>θ<sub>t</sub>d<sub>tk</sub></i></super> for their cluster "
  "<i>ω<sub>φ</sub></i>. The pieces of evidence supporting the same cluster are fused by (2), which gives")
story.append(EQ(r"m_t^{\varphi} = 1 - \prod_{x_k \in S_t,\ \ell_k = \varphi} \left(1 - e^{-\theta_t d_{tk}}\right)."))
P("The clusters close to the strongest one form the adaptive frame", body0)
story.append(EQ(r"M_t = \{\omega_{\varphi} : \max_{\psi} m_t^{\psi} - m_t^{\varphi} \leq \zeta\},"))
P(f"and every <i>A</i> ⊆ {Mt} receives the mass", body0)
story.append(EQ(r"m^{M_t}(A) = \prod_{\omega_{\varphi} \in A} m_t^{\varphi} \prod_{\omega_{\varphi} \in M_t \backslash A} \left(1 - m_t^{\varphi}\right),"))
P("where <i>A</i> = ∅ gives the noise mass. The object is assigned to the focal set with the highest normalized BBA, or to "
  "the cluster with the highest BetP (3) when a crisp decision is required.", body0)
P(f"<i>Cost:</i> The reference implementation computes and sorts the full distance matrix, i.e., "
  f"<i>O</i>({N}<super>2</super> log {N}) time and <i>O</i>({N}<super>2</super>) memory, about 20 GB at {N} = 50 000. "
  f"Only three <i>neighbour queries</i> need this matrix: the {K1} nearest distances for (5), the nearest higher-belief "
  f"object for (6), and the {K2} nearest allocated objects for (7). The quantile (4) needs all distances but not their storage.")

H2("C. Related Works")
P(f"<i>1) Evidential clustering:</i> ECM {c('masson2008')} is the evidential version of fuzzy C-means. EVCLUS "
  f"{c('denoeux2004')} derives credal partitions from proximity data, credal C-means {c('liu2015')} models imprecision "
  f"between clusters, and BPEC {c('su2019')} introduces belief peaks. These methods are partition-based, whereas BFI uses a "
  f"density-peak strategy and therefore handles arbitrary shapes {c('zhang2025')}. A recent survey of evidential "
  f"clustering is given in {c('zhang2025survey')}.")
P(f"<i>2) Efficient density-peaks clustering:</i> The quadratic cost of DPC {c('rodriguez2014')} has motivated kNN-based "
  f"formulations {c('du2016,xie2016')}, computation restricted to relevant regions {c('guan2024')}, "
  f"minimum-spanning-tree-based acceleration {c('qiu2023')}, and parallel clustering with graph-based approximate "
  f"nearest-neighbour search {c('yu2026')}. Most of these methods change the density definition or the allocation rule, "
  "or accept approximate neighbours.")
P(f"<i>3) Nearest-neighbour search:</i> Exact <i>k</i>-NN queries in low dimension are answered by k-d trees "
  f"{c('bentley1975,friedman1977')}, whose efficiency degrades in high dimension {c('weber1998')}. Approximate methods "
  f"such as NN-Descent {c('dong2011')} and HNSW {c('malkov2020')} trade exactness for speed.")
P(f"For BFI, every change of the neighbours changes the evidence: the belief values, {de}, the centres and every BBA. "
  "To our knowledge, no accelerated version of BFI has been published. Unlike the methods above, the proposed method "
  "keeps every formula of BFI and guarantees identical evidential output. We do not claim novelty for k-d-tree search or "
  "for computing exact Euclidean distances, both of which are standard.")

# ---------------- III. Proposed Method
H1(sc("III. Proposed Method"))
P("The proposed method is based on two observations. First, the three neighbour queries of BFI are <i>local</i>: for "
  "most objects the answer lies among a few nearest neighbours, so a short candidate list usually suffices. Second, it "
  "can be <i>proven</i> from a sorted exact list whether the list suffices, so exact computation is needed only for the "
  "few remaining queries. Based on these observations, the method consists of two components, presented in Sections "
  "III-A and III-B. All evidential formulas (4)–(9) are evaluated by the same code as in the original BFI.", body0)
H2("A. Solution 1: Adaptive Candidate Neighbour Retrieval")
P(f"A k-d tree is built once in <i>O</i>({N} log {N}). Every object <i>x<sub>i</sub></i> receives a <i>base list</i> "
  f"<i>L<sub>i</sub></i>: its {kmin} = 16 nearest neighbours with exact distances, sorted by distance.", body0)
P(f"<i>1) Sufficiency certificates:</i> Let <i>L<sub>i</sub></i><super>(<i>k</i>)</super> be the exact <i>k</i> nearest "
  "neighbours of <i>x<sub>i</sub></i> in increasing order of distance.")
ITEMS([f"<i>{de}-certificate:</i> If <i>L<sub>i</sub></i><super>(<i>k</i>)</super> contains an object with "
       "Bel<sub><i>j</i></sub> &gt; Bel<sub><i>i</i></sub>, the first such object is the nearest higher-belief object "
       "overall, because every object closer to <i>x<sub>i</sub></i> belongs to the list and has been checked.",
       f"<i>Convergence certificate:</i> If, when <i>x<sub>t</sub></i> is processed, <i>L<sub>t</sub></i><super>(<i>k</i>)</super> "
       f"contains at least {K2} allocated objects, its first {K2} allocated entries are exactly {St}, for the same reason."],
      labels=["a)", "b)"])
P("A query answered under a certificate is therefore identical to the original's answer; it is not an approximation.", body0)
P(f"<i>2) Adaptive growth:</i> A list that fails its certificate is regrown through the index to 2<i>k</i>, 4<i>k</i>, … "
  f"up to a budget {kmax} = 512. The candidate size thus adapts to the local data structure. Cluster cores are resolved "
  "by the base list, while density peaks, cluster borders, sparse regions and objects allocated early in the convergence, "
  "when few objects are allocated, search further. Queries still uncertified at the budget are passed to Solution 2. The "
  "procedure is summarized in Algorithm 1.")
algorithm(1, "Adaptive Candidate Neighbour Retrieval.",
          "Data <i>X</i> = {<i>x</i><sub>1</sub>, …, <i>x<sub>N</sub></i>}; belief ranks; base lists <i>L<sub>i</sub></i>.",
          f"{kmin}, {kmax}, {K2}.",
          [(0, f"<b>for</b> <i>i</i> in 1 : {N} <b>do</b>"),
           (1, f"<i>k</i> ← {kmin};"),
           (1, f"<b>while</b> <i>L<sub>i</sub></i><super>(<i>k</i>)</super> has no higher-Bel object <b>and</b> <i>k</i> &lt; {kmax} <b>do</b>"),
           (2, "<i>k</i> ← 2<i>k</i>; obtain <i>L<sub>i</sub></i><super>(<i>k</i>)</super> from the k-d tree;"),
           (1, f"<b>if</b> certified <b>then</b> {de}<sub><i>i</i></sub> ← distance to first higher-Bel entry;"),
           (1, "<b>else</b> mark <i>x<sub>i</sub></i> for exact refinement (Algorithm 2);"),
           (0, "<b>end for</b>"),
           (0, f"Select the <i>c</i> centres by {ga}<sub><i>i</i></sub>;"),
           (0, "<b>for</b> <i>x<sub>t</sub></i> in decreasing order of Bel <b>do</b>"),
           (1, f"Grow <i>L<sub>t</sub></i><super>(<i>k</i>)</super> until it holds {K2} allocated objects;"),
           (1, f"<b>if</b> certified <b>then</b> {St} ← first {K2} allocated entries;"),
           (1, "<b>else</b> mark <i>x<sub>t</sub></i> for exact refinement (Algorithm 2);"),
           (1, "Fuse evidence by (7)–(9) and allocate <i>x<sub>t</sub></i>;"),
           (0, "<b>end for</b>"),
           (0, "<b>Return:</b> BBAs, BetP, clustering results.")])

H2("B. Solution 2: Exact Distance-Based Refinement")
P(f"<i>1) Exact belief, streamed:</i> The exact {K1} nearest distances are obtained from the index for chunks of objects "
  "(at most 2<super>19</super> entries per chunk). They are reduced immediately to the log-product of (5) and then "
  f"discarded. The first {kmin} columns are kept as the base lists of Solution 1. The memory is "
  f"<i>O</i>({N}{kmin} + chunk) instead of <i>O</i>({N}<super>2</super>).", body0)
P(f"<i>2) Restricted exact scans:</i> A {de} query not certified within {kmax} is answered by computing distances only to "
  "the rank(<i>i</i>) objects with higher belief. An uncertified convergence query is answered by computing distances "
  "only to the currently allocated objects.")
P(f"<i>3) Exact distance factor:</i> Eq. (4) is computed by chunked distance blocks and partial selection, without sorting "
  f"and in <i>O</i>({N}) memory. Optionally, {th} is estimated from a random reference sample of <i>s</i> = 1000 objects, "
  f"which reduces this step from <i>O</i>({N}<super>2</super>) to <i>O</i>({N}<i>s</i>) time. This is the <i>only</i> "
  "approximation in the practical configuration. The procedure is summarized in Algorithm 2.")
algorithm(2, "Exact Distance-Based Refinement.",
          f"Data <i>X</i>; {K1}; queries marked by Algorithm 1.",
          "chunk budget, <i>q</i>, optional sample size <i>s</i>.",
          [(0, f"Compute {th}<sub><i>i</i></sub> by (4) with chunked partial selection (or from <i>s</i> samples);"),
           (0, "<b>for</b> each chunk <i>R</i> of objects <b>do</b>"),
           (1, f"Obtain the exact {K1} nearest distances of <i>R</i> from the k-d tree;"),
           (1, f"Compute Bel<sub><i>i</i></sub> by (5); keep the first {kmin} neighbours as <i>L<sub>i</sub></i>;"),
           (0, "<b>end for</b>"),
           (0, f"<b>for</b> each marked {de} query <i>x<sub>i</sub></i> <b>do</b>"),
           (1, f"{de}<sub><i>i</i></sub> ← min distance over the higher-belief subset only;"),
           (0, "<b>end for</b>"),
           (0, "<b>for</b> each marked convergence query <i>x<sub>t</sub></i> <b>do</b>"),
           (1, f"{St} ← {K2} nearest objects among the allocated subset only;"),
           (0, "<b>end for</b>"),
           (0, f"<b>Return:</b> {th}, Bel, exact {de} and {St}.")])

H2("C. Exactness of the Proposed Method")
P(f"<i>Proposition:</i> With exact {th}, the proposed method (Solution 1 + Solution 2) returns the same Bel, {de}, centres, "
  "BBAs, pignistic probabilities and meta-clusters as the original BFI, up to floating-point rounding, for any "
  f"{kmin} ≤ {kmax}.", body0)
P(f"<i>Reason:</i> Bel uses the exact {K1} nearest distances. Every {de} and convergence query is answered either under a "
  "certificate or by an exact scan of the complete candidate subset. The evidential formulas are shared with the original. "
  f"This was verified by unit tests for several ({kmin}, {kmax}), including the degenerate {kmax} = 3.")
H2("D. Parameters Involved")
P(f"The proposed method keeps the BFI parameters {K1}, {K2}, {ze} and <i>q</i>. It adds the initial list size {kmin}, the "
  f"growth budget {kmax} and, optionally, the {th} sample size <i>s</i>. With exact {th}, {kmin} and {kmax} affect only the "
  f"speed and never the result; they were set by the ablation in Experiment 3 to {kmin} = 16 and {kmax} = 512. Two "
  "deviations from the literal BFI are applied identically to the original and to all variants:", body0)
ITEMS([f"centres are selected by {ga}<sub><i>i</i></sub> = Bel<sub><i>i</i></sub>·{de}<sub><i>i</i></sub> instead of "
       f"<i>ρ</i><sub><i>f</i>(<i>i</i>)</sub>{de}<sub><i>i</i></sub> with <i>ρ</i><sub><i>f</i>(<i>i</i>)</sub> = "
       f"e<super><i>δ<sub>i</sub></i></super>, because the latter is monotone in {de}<sub><i>i</i></sub> and selected "
       "isolated outliers in our tests;",
       f"{K2} = 3, because with literal Dempster fusion {K2} = 9 lets several far neighbours outweigh a near one "
       "(confirmed in Experiment 4)."])

# ---------------- IV. Experiment Applications
H1(sc("IV. Experiment Applications"))
P("This part is dedicated to assessing the efficiency and the evidence preservation of the proposed method in comparison "
  "to the original BFI across various benchmark datasets.", body0)
H2("A. Benchmark Datasets")
P("We use synthetic, high-dimensional and real-world datasets, all min-max scaled, as follows.", body0)
ITEMS([f"Six 2-D synthetic datasets with {N} = 3000 generated with scikit-learn {c('pedregosa2011')}: <i>moons</i>, "
       "<i>circles</i> and <i>anisotropic</i> (non-convex and elongated shapes); <i>blobs_separated</i> and "
       "<i>blobs_overlap</i> (separated and overlapping clusters); and <i>varied_density</i> (standard deviations 1.0, 2.5 "
       "and 0.5). A seed changes the sampled points but not the cluster layout.",
       f"Two high-dimensional synthetic datasets, <i>blobs_hd8</i> and <i>blobs_hd32</i>, with {N} = 3000, <i>d</i> = 8 and "
       "<i>d</i> = 32, and five overlapping clusters.",
       f"Five real-world datasets: <i>iris</i> ({N} = 150, <i>d</i> = 4), <i>wine</i> (178, 13), <i>breast_cancer</i> "
       "(569, 30), and handwritten <i>digits</i> with 6 classes reduced by PCA to 10 dimensions (1083, 10) or 10 classes "
       "reduced to 16 dimensions (1797, 16).",
       f"A scalability dataset: a four-component 2-D Gaussian mixture with {N} from 1000 to 50 000."])
P(f"Unless stated otherwise, {K1} = max(10, 0.06{N}), {K2} = 3, {ze} = 0.1 and <i>q</i> = 0.9.", body0)
H2("B. Evaluation Criteria")
P(f"Following {c('zhang2025')}, we use the following criteria. Except for <i>R<sub>e</sub></i> and <i>R<sub>i</sub></i>, "
  "a higher value indicates a better result.", body0)
ITEMS([f"<i>Clustering</i>: the error rate (<i>R<sub>e</sub></i>), the imprecision rate (<i>R<sub>i</sub></i>), Precision "
       "(<i>P</i>), Recall (<i>R</i>) and Rand Index (<i>RI</i>) of the crisp result, together with their evidential "
       f"versions <i>EP</i>, <i>ER</i> and <i>ERI</i> {c('zhang2025survey,liu2015')}. We use a simplified pairwise-credal "
       "variant of <i>EP</i>, <i>ER</i> and <i>ERI</i>.",
       f"<i>Agreement-adjusted</i>: ARI {c('hubert1985')}, AMI and NMI {c('strehl2002')}.",
       f"<i>Evidence preservation</i>, against the original BFI on the same data: label agreement after Hungarian "
       f"alignment {c('kuhn1955')}, the mean total-variation distance between the pignistic probabilities (BetP TV), and "
       "focal-set agreement (identical singleton or meta-cluster assignment).",
       "<i>Efficiency</i>: wall time (median of 3 runs), the additional peak memory of one run in a fresh process, and the "
       "speed-up <i>t</i><sub>orig</sub>/<i>t</i>.",
       "<i>Neighbour search</i>: candidate recall and final selection accuracy of the δ and convergence neighbours."],
      numbered=False)
H2("C. Comparison Methods")
P("Five configurations are compared in a 2×2 factorial design around the reference:", body0)
ITEMS(["<i>Original BFI</i>: the full distance matrix;",
       "<i>Control</i>: fixed 64-nearest-neighbour lists, without adaptivity or refinement, i.e., a naive kNN-BFI;",
       "<i>S1</i>: adaptive retrieval only, with uncertified queries estimated;",
       "<i>S2</i>: fixed lists with exact refinement;",
       "<i>S1+S2</i>: the proposed method."])
P(f"The accelerated variants are run with exact {th}, which isolates the neighbour search, and with {th} sampled from "
  f"<i>s</i> = 1000 objects ({DAG}), the practical setting. Five seeds are used in Experiment 1 and three in Experiments "
  "2–5; results are reported as mean ± standard deviation, without formal significance tests. All methods run "
  f"single-threaded on a 16-core Windows 11 laptop with 15.7 GB RAM (Python 3.11, NumPy {c('harris2020')}, SciPy "
  f"{c('virtanen2020')}, scikit-learn {c('pedregosa2011')}).", body0)

H2("D. Results and Analysis")
P("Within this part, we present and examine the results. Experiment 1 evaluates clustering quality and evidence "
  "preservation on the 13 datasets. Experiment 2 investigates scalability. Experiment 3 analyses the contribution of each "
  "component. Experiment 4 studies parameter sensitivity, and Experiment 5 reports a negative result.", body0)
figure(1, "exp1_clusters.png", f"Clustering results by the original BFI (left) and {DAG} (right) on the six 2-D datasets "
       f"({N} = 3000). Red stars: cluster centres; black crosses: objects assigned to meta-clusters.", max_h=560)
story.append(NextPageTemplate("syn"))
P("<i>1) Experiment 1:</i> Tables I and II display the outcomes of the methods on the synthetic and real-world datasets. "
  "Table III summarizes the agreement with the original BFI, and Fig. 1 shows the clustering results on the six 2-D "
  "datasets. The following observations can be made.")
ITEMS([f"With exact {th}, S2 and S1+S2 reproduce the original BFI on every dataset and seed. All eleven criteria in Tables "
       "I and II are identical, the label agreement is 1.000 ± 0.000, the BetP distance is 0, and the focal sets, "
       "including all meta-cluster assignments, coincide (Table III). The evidential output of BFI is therefore fully preserved.",
       "Without refinement, the clustering changes. On <i>moons</i>, ARI drops from 0.998 to 0.530 for the control and to "
       "0.822 for S1. On iris, wine and the high-dimensional blobs, the control instead <i>raises</i> ARI (e.g., wine "
       "0.443 → 0.835). However, it agrees with the original on only 83.3% of the objects on average, so this is a "
       "different clustering produced by approximation, not a preserved BFI result.",
       f"With sampled {th} ({DAG}), the ARI stays within ±0.003 of the original on the six 2-D datasets. It decreases on "
       "the four high-dimensional datasets (e.g., digits6 ARI 0.546 → 0.413), where BFI's sequential convergence "
       f"amplifies small perturbations of {th}.",
       "<i>circles</i> and <i>breast_cancer</i> have ARI ≈ 0 for every method, including the original. This is a "
       "limitation of the density-peak centre selection, not of the acceleration."],
      labels=["a)", "b)", "c)", "d)"])

agree = [["Methods", f"{th}", "ARI", "NMI", "Label agr.", "BetP TV", "Focal agr."],
         ["Original BFI", "exact", "0.630", "0.674", "1", "0", "1"],
         ["Control", "exact", "0.611", "0.639", "0.833", "0.167", "0.826"],
         ["S1", "exact", "0.636", "0.667", "0.959", "0.041", "0.955"],
         ["S2", "exact", "0.630", "0.674", "1.000", "0.000", "1.000"],
         ["S1+S2", "exact", "0.630", "0.674", "1.000", "0.000", "1.000"],
         [Paragraph(DAG, cell), "sampled", "0.605", "0.661", "0.956", "0.044", "0.955"]]
agree[0] = [Paragraph(x, cell) for x in agree[0]]
story.append(KeepTogether(table_block(3, "Agreement with the original BFI (mean over the 13 datasets)",
                                      booktabs(agree, [52, 30, 30, 30, 38, 34, 36],
                                               extra=[("FONTNAME", (2, 4), (-1, 5), "TNR-B")]))))

P(f"Table IV reports the runtime per dataset. On the 2-D datasets, {DAG} is 1.98–2.29× faster than the original "
  f"(1.44–1.67× with exact {th}), and peak memory falls from 206 MB to 18–19 MB. On blobs_hd32 and on the small real "
  "datasets there is no speed-up (Section V).")
P("The short initial lists of 16 entries already contain 99.2% of the true δ-neighbours and 98.2% of the true convergence "
  "neighbours. Growth and refinement handle the rest, so the final selection accuracy of S1+S2 is 1.000 (Fig. 2).")

eff = [[Paragraph(x, cell) for x in ["Dataset", f"{N}", "<i>d</i>", "<i>t</i> orig. (s)", f"<i>t</i> {DAG} (s)",
                                      "Speed-up (exact)", "Speed-up (†)", "Mem. orig. (MB)", f"Mem. {DAG} (MB)"]]]
for ds in SYN + REAL:
    o, e, p = summ.loc[(ds, "original", "exact")], summ.loc[(ds, "S1+S2", "exact")], summ.loc[(ds, "S1+S2", "sampled")]
    eff.append([NAMES[ds], int(o.N), int(o.dim), f"{o.time_s_mean:.3f}", f"{p.time_s_mean:.3f}", f"{e.speedup_mean:.2f}",
                f"{p.speedup_mean:.2f}", f"{o.peak_mem_mb_mean:.0f}", f"{p.peak_mem_mb_mean:.0f}"])
story.append(KeepTogether(table_block(4, "Runtime, speed-up and peak memory per dataset (mean over seeds)",
                                      booktabs(eff, [36, 22, 13, 28, 34, 30, 30, 29, 30]),
                                      f"†: {th} sampled from 1000 objects; “exact”: S1+S2 with exact {th}. Memory = additional "
                                      "peak resident memory of one run in a fresh process.")))
figure(2, "exp1_neighbour_accuracy.png", f"Neighbour-search accuracy of the compared methods (exact {th}, 13 datasets).")

P(f"<i>2) Experiment 2:</i> This study evaluates the scalability of the methods on the Gaussian mixture with {N} = 1000 to "
  "50 000 (Table V, Fig. 3). Only 0.47 GB of RAM was free during this run, so the original BFI was run up to "
  f"{N} = 4000; larger sizes would have swapped and distorted its timing. The memory of the original grows by a factor of "
  f"3.7–4.0 per doubling of {N}, i.e., quadratically. The memory of S1+S2 grows only from 12 to 38 MB over a 50-fold "
  f"increase of {N}. At {N} = 4000, peak memory falls from 366 MB to 26 MB (14×), and the runtime from 0.401 s to "
  f"0.173 s (2.31×). The speed-up grows with {N}. At {N} = 50 000, S1+S2 needs 14.6 s and 38 MB, whereas the distance "
  "matrix alone would need about 20 GB. The label agreement with the original remains ≥ 0.999, and ARI stays at "
  f"0.917–0.924 for every {N}.")
P(f"S1+S2 needs about eight exact convergence scans per run at every {N} from 2000 to 50 000. Without refinement, the "
  f"number of estimated queries grows with {N} (control: 12 to 1951; S1: 3 to 142; Fig. 4). As a result, at {N} = 50 000 "
  "the ARI falls to 0.265 for the control and 0.396 for S1, whereas S2 and S1+S2 reach 0.924.")
scal = [[Paragraph(x, cell) for x in [f"{N}", "Original time / memory", f"{DAG} time / memory", "Speed-up", "Agreement", "ARI"]],
        ["1 000", "0.037 s / 25 MB", "0.031 s / 12 MB", "1.18", "1.000", "0.917"],
        ["2 000", "0.116 s / 92 MB", "0.070 s / 9 MB", "1.67", "0.999", "0.917"],
        ["4 000", "0.401 s / 366 MB", "0.173 s / 26 MB", "2.31", "0.999", "0.917"],
        ["10 000", "–", "0.742 s / 28 MB", "–", "–", "0.919"],
        ["20 000", "–", "2.53 s / 30 MB", "–", "–", "0.923"],
        ["50 000", "(≈ 20 GB)", "14.6 s / 38 MB", "–", "–", "0.924"]]
story.append(KeepTogether(table_block(5, "Scalability on the Gaussian mixture (mean of 3 seeds)",
                                      booktabs(scal, [30, 62, 62, 30, 36, 28], extra=[("FONTNAME", (3, 3), (3, 3), "TNR-B")]))))
figure(3, "exp2_scalability.png", f"Runtime, peak memory, speed-up and time breakdown of {DAG} versus {N}.")
figure(4, "exp2_refinement_work.png", "Exact scans needed by refinement (left) and queries answered by an estimate when "
       "refinement is disabled (right).")

P(f"<i>3) Experiment 3:</i> This study analyses the contribution of each component on four datasets (varied_density, "
  f"blobs_overlap, moons, blobs_hd8) at {N} = 4000, with three seeds and exact {th} (Table VI, Fig. 5).")
ITEMS(["<i>Effect of Solution 2:</i> Refinement provides exactness. The agreement rises from 0.835 (control) or 0.914 (S1) "
       "to 1.000, at a cost of about 20% runtime at this size.",
       "<i>Effect of Solution 1:</i> Adaptive retrieval reduces the mean convergence list from 64 to about 18 entries. It "
       "gives 4× fewer estimated queries without refinement (56.3 to 13.8) and 38% fewer exact scans with refinement "
       "(12.3 to 7.6), at equal speed within the noise.",
       f"<i>Parameters:</i> {kmin} = 4, 8, 16, 32 and 64 give speed-ups of 1.11, 1.35, 1.43, 1.43 and 1.46 and are always "
       f"exact; {kmin} = 16 was chosen. For {kmax} from 32 to 1024 the speed of S1+S2 is flat (1.44–1.48) while the number "
       f"of scans falls from 19.3 to 4.2; for S1 alone, agreement peaks at 0.98 with {kmax} = 512. Refining all belief "
       "values is the only exact policy (flagged objects only: 0.949; centre contenders only: 0.893). <i>s</i> = 1000 gives "
       "agreement 0.997 at a 2.15× speed-up."], labels=["a)", "b)", "c)"])
P("In summary, <i>refinement guarantees exactness and adaptive retrieval makes refinement cheap</i>.")
fac = [[Paragraph(x, cell) for x in ["Methods", "Speed-up", "Agreement", "Conv. scans", "Estimated queries", "Mean conv. list"]],
       ["Control", "1.71 ± 0.25", "0.835", "0", "56.3", "64.0"],
       ["S1", "1.61 ± 0.25", "0.914", "0", "13.8", "20.2"],
       ["S2", "1.34 ± 0.25", "1.000", "12.3", "0", "66.5"],
       ["S1+S2", "1.37 ± 0.21", "1.000", "7.6", "0", "17.9"]]
story.append(KeepTogether(table_block(6, f"2×2 factorial ablation (mean over 12 dataset–seed pairs, {N} = 4000)",
                                      booktabs(fac, [38, 46, 40, 36, 44, 44],
                                               extra=[("FONTNAME", (2, 3), (2, 4), "TNR-B"),
                                                      ("FONTNAME", (3, 4), (3, 4), "TNR-B"),
                                                      ("FONTNAME", (5, 4), (5, 4), "TNR-B")]))))
figure(5, "exp3_factorial.png", "Factorial ablation: speed-up, agreement with the original and refinement work of the "
       "four configurations.")

P(f"<i>4) Experiment 4:</i> This study investigates the parameter sensitivity over {K1} ∈ [0.02{N}, 0.15{N}], "
  f"{K2} ∈ {{1, 3, 5, 9, 15}} and {ze} ∈ [0, 0.5], on four datasets with three seeds and sampled {th}. S1+S2 follows the "
  f"original across the whole range. Its agreement is 0.977–0.999 for {K2} ≤ 5, and 0.90 at {K2} ≥ 9, where the original "
  f"itself collapses (ARI 0.302 and 0.081). The control drops to 45–62% agreement. As in {c('zhang2025')}, {ze} regulates "
  "the number of objects assigned to meta-clusters. The imprecision rate <i>R<sub>i</sub></i> rises from 0 to 0.042 for "
  "the original and to 0.040 for S1+S2, while <i>R<sub>e</sub></i> falls from 0.150 to 0.137 and from 0.165 to 0.153, "
  "respectively.")
P("<i>5) Experiment 5:</i> We also tried to choose the belief candidate count per object from local data, using a "
  "two-scale intrinsic-dimension test or an out-of-sample validated power-law tail, so that the "
  f"{K1}-neighbour tail could be extrapolated from a short list. At equal mean candidate count, both rules were "
  "<i>worse</i> than a fixed size (mean relative belief error 0.106 with 66 candidates against 0.054 for a fixed list of "
  "64). The head of a neighbour list carries no information about density changes beyond its radius. Adaptivity is "
  "therefore used only where an exact certificate exists, and belief is computed exactly by Solution 2.")

# ---------------- V. Discussions
H1(sc("V. Discussions"))
P("In this section, we discuss the complexity and the limitations of the proposed method.", body0)
H2("A. Complexity Analysis")
P(f"Table VII compares the cost of each step. Building the index costs <i>O</i>({N} log {N}). The belief pass makes {N} "
  f"exact {K1}-NN queries (expected <i>O</i>({N}{K1} log {N}) in low dimension) with <i>O</i>({N}{kmin}) memory. The {de} "
  f"and convergence queries cost <i>O</i>({N}{kmin}) for certified objects, plus growth and scans for the few others. At "
  f"{N} = 50 000, {de} takes 0.01 s and the convergence 0.71 s; both stay cheap at every size.", body0)
cx = [[Paragraph(x, cell) for x in ["Step", "Original BFI", "Proposed (S1+S2)"]],
      [Paragraph(f"{th}", cell), Paragraph(f"<i>O</i>({N}<super>2</super>)", cell), Paragraph(f"<i>O</i>({N}<super>2</super>) exact / <i>O</i>({N}<i>s</i>) sampled", cell)],
      ["Bel", Paragraph(f"<i>O</i>({N}<super>2</super> log {N}) (sort)", cell), Paragraph(f"<i>O</i>({N}{K1} log {N})", cell)],
      [Paragraph(f"{de}", cell), Paragraph(f"<i>O</i>({N}<super>2</super>)", cell), Paragraph(f"<i>O</i>({N}{kmin}) + few scans", cell)],
      ["Convergence", Paragraph(f"<i>O</i>({N}<super>2</super>)", cell), Paragraph(f"<i>O</i>({N}{kmin}) + few scans", cell)],
      ["Memory", Paragraph(f"<i>O</i>({N}<super>2</super>)", cell), Paragraph(f"<i>O</i>({N}{kmin}) + chunk", cell)]]
story.append(KeepTogether(table_block(7, "Cost of each step (low dimension, expected k-d-tree behaviour)",
                                      booktabs(cx, [50, 80, 110]))))
H2("B. Limitations")
ITEMS([f"<i>Exact belief dominates at large {N}:</i> With {K1} = 0.06{N}, the belief pass needs <i>O</i>({N}<super>2</super>) "
       f"distance terms; at {N} = 50 000 it takes 13.5 s of the 14.6 s. The quadratic memory and the sort are removed, but "
       "not this term.",
       f"<i>Exact {th} is inherently <i>O</i>({N}<super>2</super>):</i> At {N} ≤ 4000, the exact-{th} configuration is only "
       "1.2–1.5× faster than the original. Sampling removes this cost but is the only approximation, and it changes the "
       "results on the digits datasets.",
       f"<i>High dimension and small datasets:</i> On blobs_hd32, the method is slower than the original (0.61–0.75×), because "
       f"k-d trees degrade towards brute force {c('weber1998')}. On real datasets with {N} ≤ 1800, it reaches 0.78–1.03×. "
       "The memory saving remains.",
       "<i>Scope:</i> The evaluation covers a single machine and a single thread. The Olivetti faces and Berkeley "
       f"segmentation images of {c('zhang2025')} were not used."])

# ---------------- VI. Conclusion
H1(sc("VI. Conclusion"))
P("This article introduces an efficient and evidence-preserving framework for belief-based fuzzy and imprecise "
  "clustering that changes only how neighbours are retrieved. The adaptive candidate neighbour retrieval answers the δ and "
  "evidential-convergence queries of BFI exactly from short k-d-tree lists, using sufficiency certificates, and enlarges "
  "only the lists that need it. The exact distance-based refinement streams exact belief values in bounded memory and "
  "resolves the few uncertified queries by restricted exact scans. Experimental findings on 13 datasets and on data with "
  "up to 50 000 objects show three results. First, the method reproduces the original BFI output exactly, including belief "
  "values, centres, BBAs and meta-clusters. Second, it reduces memory from quadratic to near-linear (366 MB to 26 MB at "
  f"{N} = 4000; 38 MB at {N} = 50 000). Third, it is up to 2.3× faster. The ablation shows that both components are "
  f"necessary. The method does not help on high-dimensional or small datasets, and at very large {N} the exact belief "
  "computation dominates the runtime. In the future, we will work on a certified bound on the belief tail, a parallel "
  f"convergence loop, approximate nearest-neighbour back-ends with recall guarantees for high-dimensional data "
  f"{c('malkov2020')}, and the image datasets of the original study.", body0)

# ---------------- References
H1(sc("References"))
for i, k in enumerate(ORDER):
    story.append(Paragraph(f"[{i + 1}]&nbsp;&nbsp;{REFS[k]}", ref_st))

doc.build(story)
print("saved", OUT, "| references:", len(ORDER), "| equations:", _eq_n[0])
