"""
Filix Medtech – NIR Spectrum Viewer (User App)
Main page: https://filix-medtech-heaudbkmpj29m6ytihv8bg.streamlit.app
Admin page: https://filix-medtech-heaudbkmpj29m6ytihv8bg.streamlit.app/Admin
"""

import streamlit as st
import pandas as pd
import numpy as np
from numpy.polynomial import legendre
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, io, datetime, json, base64, re
import requests

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH      = os.path.join(BASE_DIR, "filix_logo.png")
MEDICINES_JSON = os.path.join(BASE_DIR, "medicines.json")
MEDICINES_DATA = os.path.join(BASE_DIR, "medicines_data")

# ─────────────────────────────────────────────
#  GitHub API helpers
# ─────────────────────────────────────────────
GITHUB_REPO   = "kqilq/filix-medtech"
GITHUB_BRANCH = "main"
DATA_FOLDER   = "medicines_data"

def _gh_token():
    try:    return st.secrets["GITHUB_TOKEN"]
    except: return os.environ.get("GITHUB_TOKEN", "")

def _gh_headers():
    t = _gh_token()
    h = {"Accept": "application/vnd.github.v3+json"}
    if t: h["Authorization"] = f"token {t}"
    return h

def _gh_api(path):
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

def gh_read_json_medicines():
    r = requests.get(_gh_api("medicines.json"), headers=_gh_headers(),
                     params={"ref": GITHUB_BRANCH})
    if r.status_code != 200: return None
    return json.loads(base64.b64decode(r.json()["content"]).decode()).get("medicines", [])

def gh_read_csv_bytes(filename):
    for path in [f"{DATA_FOLDER}/{filename}", filename]:
        r = requests.get(_gh_api(path), headers=_gh_headers(),
                         params={"ref": GITHUB_BRANCH})
        if r.status_code == 200:
            return base64.b64decode(r.json()["content"])
    return None

# ─────────────────────────────────────────────
#  Load medicines
# ─────────────────────────────────────────────
def load_medicines():
    try:
        entries = gh_read_json_medicines()
        if entries is not None:
            return {e["name"]: {
                "csv": e["csv"], "description": e.get("description",""),
                "description_zh": e.get("description_zh",""), "from_github": True,
            } for e in entries}
    except: pass
    if not os.path.exists(MEDICINES_JSON): return {}
    with open(MEDICINES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for e in data.get("medicines", []):
        fn = e["csv"]
        if os.path.isabs(fn): p = fn
        elif os.path.exists(os.path.join(BASE_DIR, fn)): p = os.path.join(BASE_DIR, fn)
        elif os.path.exists(os.path.join(MEDICINES_DATA, fn)): p = os.path.join(MEDICINES_DATA, fn)
        else: p = os.path.join(BASE_DIR, fn)
        result[e["name"]] = {"csv": p, "description": e.get("description",""),
                              "description_zh": e.get("description_zh",""), "from_github": False}
    return result

def _get_csv_bytes(med_info):
    if med_info.get("from_github"):
        data = gh_read_csv_bytes(med_info["csv"])
        if data is None: raise FileNotFoundError(f"CSV not found: {med_info['csv']}")
        return data
    with open(med_info["csv"], "rb") as f: return f.read()

# ─────────────────────────────────────────────
#  Localisation
# ─────────────────────────────────────────────
STRINGS = {
    "en": {
        "app_title": "Filix Medtech – NIR Spectrum Viewer",
        "list_title": "List of Medicines",
        "list_subtitle": "Select a medicine to view its spectrum and compare with your sample.",
        "view_spectrum": "View Medicine Info",
        "back_list": "← Back to List",
        "ref_spectrum": "Reference Spectrum",
        "upload_section": "Compare with your CSV",
        "upload_note": "ℹ️ Please upload a CSV file exported from LSCollector.",
        "upload_btn": "Upload CSV",
        "clear_btn": "Clear uploaded file",
        "analysis_title": "Analysis Results",
        "degree": "Degree", "adj_r2": "Adj. R²", "r2": "R²", "ratio": "Adj.R²/R² (%)",
        "chosen_degree": "Chosen best degree", "best_ratio": "Best degree ratio to 100%",
        "self_acc": "Self accuracy (R²)", "test_acc": "Test accuracy (R²)",
        "similarity": "Similarity", "similarity_result": "🎯 Similarity",
        "how_to_read": "ℹ️ How to read these results",
        "running": "⏳ Running analysis…", "language_btn": "繁體中文",
        "explanations": [
            ("📐 Polynomial Degree",
             "The spectrum curve is fitted using a polynomial equation. A higher degree means a more complex curve. "
             "The table shows all degrees that gave statistically significant fits (p < 0.001 for all terms)."),
            ("📊 Adjusted R² (Adj. R²)",
             "Measures how well the polynomial fits the reference spectrum, penalising unnecessary complexity. "
             "Closer to 1.0 = better fit. This is used to select the best degree."),
            ("📈 R² (R-squared)",
             "The basic goodness-of-fit score — the proportion of variance in the spectrum explained by the polynomial. "
             "1.0 = perfect fit, 0 = no fit at all."),
            ("⚖️ Adj. R² / R² (%)",
             "The ratio of Adjusted R² to R², expressed as a percentage. The best degree is the one where this ratio "
             "is closest to 100% — meaning the model is neither over-fitted nor under-fitted."),
            ("🏆 Chosen Best Degree",
             "The polynomial degree whose Adj. R²/R² ratio is closest to 100%. This is the most balanced model — "
             "complex enough to capture the spectrum shape, but not so complex that it overfits noise."),
            ("🔬 Self Accuracy (R²)",
             "How well the chosen polynomial model fits the reference medicine's own spectrum. "
             "This is the baseline — ideally very close to 1.0."),
            ("🧪 Test Accuracy (R²)",
             "How well the same model fits your uploaded sample's spectrum. A value close to the Self Accuracy "
             "means your sample behaves similarly to the reference."),
            ("🎯 Similarity (%)",
             "Test Accuracy ÷ Self Accuracy × 100. This is the key result:\n"
             "- ~100% → your sample is very similar to the reference medicine\n"
             "- Much lower → the spectra differ significantly\n"
             "- Can exceed 100% if your sample fits the model even better than the reference"),
        ],
    },
    "zh": {
        "app_title": "Filix Medtech – 近紅外光譜檢視器",
        "list_title": "藥品列表",
        "list_subtitle": "選擇藥品以查看其光譜並與您的樣本比較。",
        "view_spectrum": "查看藥品資訊", "back_list": "← 返回列表",
        "ref_spectrum": "參考光譜", "upload_section": "與您的 CSV 比較",
        "upload_note": "ℹ️ 請上傳從 LSCollector 匯出的 CSV 檔案。",
        "upload_btn": "上傳 CSV", "clear_btn": "清除已上傳檔案",
        "analysis_title": "分析結果", "degree": "階數", "adj_r2": "Adj. R²",
        "r2": "R²", "ratio": "Adj.R²/R² (%)", "chosen_degree": "最佳階數",
        "best_ratio": "最佳階數與 100% 的比率", "self_acc": "自身準確度 (R²)",
        "test_acc": "測試準確度 (R²)", "similarity": "相似度",
        "similarity_result": "🎯 相似度", "how_to_read": "ℹ️ 如何解讀分析結果",
        "running": "⏳ 正在分析中…", "language_btn": "English",
        "explanations": [
            ("📐 多項式階數（Polynomial Degree）",
             "光譜曲線以多項式方程式進行擬合。階數越高，曲線越複雜。"
             "表格顯示所有統計上顯著的階數（所有項目 p < 0.001）。"),
            ("📊 調整後 R²（Adj. R²）",
             "衡量多項式對參考光譜的擬合程度，並對不必要的複雜度進行懲罰。"
             "越接近 1.0 表示擬合越好。此數值用於選擇最佳階數。"),
            ("📈 R²（決定係數）",
             "基本擬合優度分數——多項式能解釋光譜變異的比例。"
             "1.0 = 完美擬合，0 = 完全無法擬合。"),
            ("⚖️ Adj. R² / R² (%)",
             "Adj. R² 與 R² 的比率，以百分比表示。"
             "最佳階數是此比率最接近 100% 的那個——代表模型既不過度擬合也不欠擬合。"),
            ("🏆 最佳階數",
             "Adj. R²/R² 比率最接近 100% 的多項式階數。"
             "這是最平衡的模型——足夠複雜以捕捉光譜形狀，但不會過度擬合雜訊。"),
            ("🔬 自身準確度（R²）",
             "所選多項式模型對參考藥品本身光譜的擬合程度。"
             "這是基準值——理想情況下應非常接近 1.0。"),
            ("🧪 測試準確度（R²）",
             "相同模型對您上傳樣本光譜的擬合程度。"
             "數值越接近自身準確度，表示您的樣本與參考藥品越相似。"),
            ("🎯 相似度（%）",
             "測試準確度 ÷ 自身準確度 × 100。這是最關鍵的結果：\n"
             "- 約 100% → 您的樣本與參考藥品非常相似\n"
             "- 遠低於 100% → 光譜差異顯著\n"
             "- 可能超過 100%（若您的樣本比參考藥品更符合模型）"),
        ],
    },
}

# ─────────────────────────────────────────────
#  Analysis helpers
# ─────────────────────────────────────────────
def clean_file(b):
    df = pd.read_csv(io.BytesIO(b), index_col=0)
    df_T = df.transpose()
    mi = df_T.mean(axis=1, skipna=True)
    return pd.DataFrame({"Wavelength": df_T.index.astype(float), "Intensity": mi.values}).sort_values("Wavelength")

def load_spectrum(b):
    df = pd.read_csv(io.BytesIO(b), header=0)
    iv = df.iloc[0, 1:].astype(float).values
    return list(range(len(iv))), iv

def legendre_features(x, degree):
    xs = 2*(x-np.min(x))/(np.max(x)-np.min(x))-1
    return legendre.legvander(xs, degree)

def run_analysis(std_bytes, test_bytes):
    standard = clean_file(std_bytes)
    test     = clean_file(test_bytes)
    wl = standard["Wavelength"].values
    intensity = standard["Intensity"].values
    valid_degrees, adj_r2s, stop_found = [], [], False
    for degree in range(2, 11):
        XX = sm.add_constant(legendre_features(wl, degree)[:, 1:])
        model = sm.OLS(intensity, XX).fit()
        if np.all(model.pvalues[1:] < 0.001) and not stop_found:
            valid_degrees.append(degree); adj_r2s.append(model.rsquared_adj)
        elif not stop_found:
            stop_found = True
    if not valid_degrees: raise ValueError("No valid polynomial degrees found.")
    X = standard[["Wavelength"]].values; y = standard["Intensity"].values
    r_square, ratios = [], []
    for idx, degree in enumerate(valid_degrees):
        pr = PolynomialFeatures(degree=degree); Xp = pr.fit_transform(X)
        res = sm.OLS(y, Xp).fit()
        RSS = TSS = 0
        for i in range(len(y)):
            ye = res.predict(pr.fit_transform([[X[i][0]]]))[0]
            RSS += (y[i]-ye)**2; TSS += (y[i]-np.mean(y))**2
        R_sq = 1-RSS/TSS; r_square.append(R_sq)
        ratios.append((adj_r2s[idx]/R_sq)*100)
    bi = int(np.argmin(np.abs(np.array(ratios)-100)))
    bd = valid_degrees[bi]
    pr = PolynomialFeatures(degree=bd); Xp = pr.fit_transform(X)
    fm = sm.OLS(y, Xp).fit()
    RSS = TSS = 0
    for i in range(len(y)):
        ye = fm.predict(pr.fit_transform([[X[i][0]]]))[0]
        RSS += (y[i]-ye)**2; TSS += (y[i]-np.mean(y))**2
    fR = 1-RSS/TSS
    yt = test["Intensity"].values
    pr = PolynomialFeatures(degree=bd); Xp = pr.fit_transform(X)
    res = sm.OLS(y, Xp).fit()
    RSS = TSS = 0
    for i in range(len(y)):
        ye = res.predict(pr.fit_transform([[X[i][0]]]))[0]
        RSS += (yt[i]-ye)**2; TSS += (yt[i]-np.mean(y))**2
    Rt = 1-RSS/TSS
    return {
        "table": [{"degree": int(d), "Adj. R²": round(a,6), "R²": round(r,6),
                   "Adj.R²/R² (%)": round(rt,4)}
                  for d,a,r,rt in zip(valid_degrees,adj_r2s,r_square,ratios)],
        "best_degree": bd, "best_ratio": ratios[bi],
        "self_acc": fR, "test_acc": Rt, "similarity": (Rt/fR)*100,
    }

def make_fig(px, intensity, title, color):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(px, intensity, color=color, linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Pixel Number", fontsize=11, labelpad=6)
    ax.set_ylabel("Intensity (counts)", fontsize=11, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); return fig

def fig_to_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0); return buf.read()

def _pdf_safe(text):
    for u,a in {"\u2013":"-","\u2014":"--","\u2192":"->","\u00f7":"/",
                "\u00d7":"x","\u2019":"'","\u2018":"'","\u201c":'"',"\u201d":'"',
                "\u2026":"...","\u00b2":"2"}.items():
        text = text.replace(u,a)
    return text.encode("latin-1", errors="replace").decode("latin-1")

_ARIAL_PATH = os.path.join(BASE_DIR, "fonts", "ArialUnicode.ttf")
_ARIAL_REG  = False
def _ensure_arial():
    global _ARIAL_REG
    if not _ARIAL_REG and os.path.exists(_ARIAL_PATH):
        pdfmetrics.registerFont(TTFont("ArialUnicode", _ARIAL_PATH)); _ARIAL_REG = True
    return _ARIAL_REG

PDF_EXP_EN = [
    ("Polynomial Degree","The spectrum curve is fitted using a polynomial equation. A higher degree means a more complex curve. The table shows all degrees that gave statistically significant fits (p < 0.001 for all terms)."),
    ("Adjusted R2 (Adj. R2)","Measures how well the polynomial fits the reference spectrum, penalising unnecessary complexity. Closer to 1.0 = better fit."),
    ("R2 (R-squared)","The basic goodness-of-fit score - the proportion of variance in the spectrum explained by the polynomial. 1.0 = perfect fit, 0 = no fit at all."),
    ("Adj. R2 / R2 (%)","The ratio of Adjusted R2 to R2, expressed as a percentage. The best degree is the one where this ratio is closest to 100%."),
    ("Chosen Best Degree","The polynomial degree whose Adj. R2/R2 ratio is closest to 100%. This is the most balanced model."),
    ("Self Accuracy (R2)","How well the chosen polynomial model fits the reference medicine's own spectrum. This is the baseline - ideally very close to 1.0."),
    ("Test Accuracy (R2)","How well the same model fits your uploaded sample's spectrum. A value close to the Self Accuracy means your sample behaves similarly to the reference."),
    ("Similarity (%)","Test Accuracy / Self Accuracy x 100. ~100%: very similar. Much lower: spectra differ significantly."),
]
PDF_EXP_ZH = [
    ("多項式階數","光譜曲線以多項式方程式進行擬合。階數越高，曲線越複雜。"),
    ("調整後 R2","衡量多項式對參考光譜的擬合程度，越接近 1.0 表示擬合越好。"),
    ("R2（決定係數）","基本擬合優度分數——多項式能解釋光譜變異的比例。1.0 = 完美擬合。"),
    ("Adj. R2 / R2 (%)","最佳階數是此比率最接近 100% 的那個——代表模型既不過度擬合也不欠擬合。"),
    ("最佳階數","Adj. R2/R2 比率最接近 100% 的多項式階數。"),
    ("自身準確度","所選多項式模型對參考藥品本身光譜的擬合程度。"),
    ("測試準確度","相同模型對您上傳樣本光譜的擬合程度。"),
    ("相似度（%）","測試準確度 / 自身準確度 x 100。約 100%：非常相似。"),
]
PDF_ZH = {
    "report_title":"近紅外光譜分析報告","ref_medicine":"參考藥品：","date":"日期：",
    "sample_file":"樣本檔案：","spectrum_graphs":"光譜圖","ref_spectrum":"參考光譜 - ",
    "sample_spectrum":"樣本光譜 - ","analysis":"分析結果","degree":"階數",
    "adj_r2":"Adj. R2","r2":"R2","ratio":"Adj.R2/R2 (%)","best_degree":"最佳階數",
    "best_ratio":"最佳階數比率","self_acc":"自身準確度 (R2)","test_acc":"測試準確度 (R2)",
    "similarity":"相似度","sim_label":"相似度：","how_to_read":"如何解讀分析結果",
    "footer":"由 Filix Medtech 近紅外光譜檢視器生成",
}

def generate_pdf(med_name, sample_name, fig_ref_bytes, fig_smp_bytes, res, lang="en"):
    buf = io.BytesIO()
    use_zh = (lang=="zh") and _ensure_arial()
    FR = "ArialUnicode" if use_zh else "Helvetica"
    FB = "ArialUnicode" if use_zh else "Helvetica-Bold"
    ZH = PDF_ZH if use_zh else {}
    PURPLE=colors.HexColor("#3a3a5c"); LB=colors.HexColor("#eef4fb")
    BD=colors.HexColor("#b0cce8"); GBG=colors.HexColor("#f0fff0")
    GBD=colors.HexColor("#5a7a5c"); GREY=colors.HexColor("#555555"); W_=colors.white
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    W = A4[0]-4*cm
    ts = ParagraphStyle("t",fontSize=20,textColor=PURPLE,spaceAfter=4,fontName=FB,alignment=TA_CENTER)
    ss = ParagraphStyle("s",fontSize=11,textColor=GREY,spaceAfter=2,fontName=FR,alignment=TA_CENTER)
    h2 = ParagraphStyle("h2",fontSize=14,textColor=PURPLE,spaceBefore=14,spaceAfter=6,fontName=FB)
    h3 = ParagraphStyle("h3",fontSize=12,textColor=PURPLE,spaceBefore=10,spaceAfter=4,fontName=FB)
    bs = ParagraphStyle("b",fontSize=10,textColor=GREY,spaceAfter=4,fontName=FR,leading=15)
    bds= ParagraphStyle("bd",fontSize=10,textColor=PURPLE,spaceAfter=2,fontName=FB)
    sms= ParagraphStyle("sm",fontSize=9,textColor=GREY,spaceAfter=2,fontName=FR)
    story=[]
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH,width=2.5*cm,height=2.5*cm))
        story.append(Spacer(1,0.5*cm))
    story.append(Paragraph("Filix Medtech",ts))
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph(ZH.get("report_title","NIR Spectrum Analysis Report") if use_zh else "NIR Spectrum Analysis Report",ss))
    story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=2,color=PURPLE))
    story.append(Spacer(1,0.4*cm))
    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if use_zh:
        id_=[[ZH["ref_medicine"],med_name,ZH["date"],now],[ZH["sample_file"],sample_name,"",""]]
    else:
        id_=[["Reference Medicine:",med_name,"Date:",now],["Sample File:",sample_name,"",""]]
    it=Table(id_,colWidths=[3.5*cm,6*cm,2*cm,5*cm])
    it.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),FR),("FONTNAME",(0,0),(0,-1),FB),
        ("FONTNAME",(2,0),(2,-1),FB),("FONTSIZE",(0,0),(-1,-1),10),
        ("TEXTCOLOR",(0,0),(-1,-1),GREY),("TEXTCOLOR",(0,0),(0,-1),PURPLE),
        ("TEXTCOLOR",(2,0),(2,-1),PURPLE),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(it); story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=BD))
    story.append(Paragraph(ZH.get("spectrum_graphs","Spectrum Graphs") if use_zh else "Spectrum Graphs",h2))
    rl=(ZH["ref_spectrum"]+med_name) if use_zh else _pdf_safe(f"Reference Spectrum - {med_name}")
    story.append(Paragraph(rl,h3))
    story.append(RLImage(io.BytesIO(fig_ref_bytes),width=W,height=W*0.38))
    story.append(Spacer(1,0.3*cm))
    sl=(ZH["sample_spectrum"]+sample_name) if use_zh else _pdf_safe(f"Sample Spectrum - {sample_name}")
    story.append(Paragraph(sl,h3))
    story.append(RLImage(io.BytesIO(fig_smp_bytes),width=W,height=W*0.38))
    story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=BD))
    story.append(PageBreak())
    story.append(Paragraph(ZH.get("analysis","Analysis Results") if use_zh else "Analysis Results",h2))
    hdr=[ZH["degree"],ZH["adj_r2"],ZH["r2"],ZH["ratio"]] if use_zh else ["Degree","Adj. R2","R2","Adj.R2/R2 (%)"]
    td=[hdr]+[[str(r["degree"]),f"{r['Adj. R\u00b2']:.6f}",f"{r['R\u00b2']:.6f}",f"{r['Adj.R\u00b2/R\u00b2 (%)']:.4f}"] for r in res["table"]]
    cw=W/4
    dt=Table(td,colWidths=[cw]*4)
    dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),PURPLE),("TEXTCOLOR",(0,0),(-1,0),W_),
        ("FONTNAME",(0,0),(-1,0),FB),("FONTNAME",(0,1),(-1,-1),FR),("FONTSIZE",(0,0),(-1,-1),10),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[W_,LB]),
        ("GRID",(0,0),(-1,-1),0.5,BD),("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6)]))
    story.append(dt); story.append(Spacer(1,0.4*cm))
    sim=res["similarity"]
    if use_zh:
        mx=[[ZH["best_degree"],str(res["best_degree"])],[ZH["best_ratio"],f"{res['best_ratio']:.4f} %"],
            [ZH["self_acc"],f"{res['self_acc']:.6f}"],[ZH["test_acc"],f"{res['test_acc']:.6f}"],
            [ZH["similarity"],f"{sim:.4f} %"]]
    else:
        mx=[["Chosen Best Degree",str(res["best_degree"])],["Best Degree Ratio",f"{res['best_ratio']:.4f} %"],
            ["Self Accuracy (R2)",f"{res['self_acc']:.6f}"],["Test Accuracy (R2)",f"{res['test_acc']:.6f}"],
            ["Similarity",f"{sim:.4f} %"]]
    mt=Table(mx,colWidths=[W*0.5,W*0.5])
    mt.setStyle(TableStyle([("FONTNAME",(0,0),(0,-1),FB),("FONTNAME",(1,0),(1,-1),FR),
        ("FONTSIZE",(0,0),(-1,-1),10),("TEXTCOLOR",(0,0),(0,-1),PURPLE),("TEXTCOLOR",(1,0),(1,-1),GREY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[W_,LB]),("GRID",(0,0),(-1,-1),0.5,BD),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("ALIGN",(1,0),(1,-1),"RIGHT")]))
    story.append(mt); story.append(Spacer(1,0.3*cm))
    sl2=f"{ZH['sim_label']}{sim:.2f}%" if use_zh else f"Similarity: {sim:.2f}%"
    st2=Table([[sl2]],colWidths=[W])
    st2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GBG),("TEXTCOLOR",(0,0),(-1,-1),GBD),
        ("FONTNAME",(0,0),(-1,-1),FB),("FONTSIZE",(0,0),(-1,-1),16),("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("BOX",(0,0),(-1,-1),1.5,GBD),("BOTTOMPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),12)]))
    story.append(st2); story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=BD))
    story.append(PageBreak())
    story.append(Paragraph(ZH.get("how_to_read","How to Read These Results") if use_zh else "How to Read These Results",h2))
    story.append(Spacer(1,0.2*cm))
    for et,ed in (PDF_EXP_ZH if use_zh else PDF_EXP_EN):
        story.append(Paragraph(et if use_zh else _pdf_safe(et),bds))
        for line in ed.split("\n"):
            line=line.strip()
            if line: story.append(Paragraph(line if use_zh else _pdf_safe(line),bs))
        story.append(Spacer(1,0.2*cm))
    story.append(Spacer(1,0.4*cm))
    story.append(HRFlowable(width="100%",thickness=1,color=PURPLE))
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph(ZH.get("footer","Generated by Filix Medtech NIR Spectrum Viewer") if use_zh else "Generated by Filix Medtech NIR Spectrum Viewer",sms))
    doc.build(story); buf.seek(0); return buf.read()

# ─────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Filix Medtech", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    /* Hide the sidebar page navigation so users don't see the Admin link */
    [data-testid="stSidebarNav"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    html, body, [class*="css"] { font-size: 17px !important; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.7rem !important; }
    h3 { font-size: 1.4rem !important; }
    p, label, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem !important; line-height: 1.6 !important; }
    .stButton > button { font-size: 1.1rem !important; padding: 0.55rem 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .med-card { background:white; border:1px solid #ddd; border-radius:10px;
        padding:20px 24px; margin-bottom:16px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
    .med-card h3 { margin:0 0 6px 0; color:#3a3a5c; font-size:1.3rem; }
    .med-card p  { margin:0 0 12px 0; color:#555; font-size:1.05rem; }
    .info-note { background:#eef4fb; border:1px solid #b0cce8; border-radius:6px;
        padding:10px 16px; color:#2a5080; font-size:1rem; margin-bottom:12px; }
    .similarity-box { background:#f0fff0; border:2px solid #5a7a5c; border-radius:8px;
        padding:16px 24px; text-align:center; font-size:1.6rem;
        font-weight:bold; color:#3a3a5c; margin:12px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for k,v in [("lang","en"),("page","list"),("selected_med",None),
             ("upload_bytes",None),("analysis_res",None)]:
    if k not in st.session_state: st.session_state[k] = v

S = STRINGS[st.session_state.lang]
MEDICINES = load_medicines()

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
col_logo, col_title, col_lang = st.columns([0.08, 0.82, 0.10])
with col_logo:
    if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=52)
with col_title:
    st.markdown("<div style='padding-top:6px'><h2 style='color:#3a3a5c;margin:0'>Filix Medtech"
                "<span style='font-size:1rem;color:#aaa;margin-left:10px'>· NIR Spectrum Viewer</span>"
                "</h2></div>", unsafe_allow_html=True)
with col_lang:
    if st.button(S["language_btn"], key="lang_btn"):
        st.session_state.lang = "zh" if st.session_state.lang=="en" else "en"
        st.rerun()

st.markdown("<hr style='margin:4px 0 16px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIST PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "list":
    st.markdown(f"## {S['list_title']}")
    st.markdown(f"<p style='color:#555;margin-top:-8px'>{S['list_subtitle']}</p>",
                unsafe_allow_html=True)

    # ── How to Use guide ──────────────────────
    if st.session_state.lang == "en":
        with st.expander("ℹ️ New here? How to use this app", expanded=False):
            st.markdown("""
**Welcome to Filix Medtech – NIR Spectrum Viewer** 👋

This app helps you verify whether a medicine sample matches a known reference using Near-Infrared (NIR) spectroscopy.

---

**What you need before you start:**
- A CSV file exported from the **LSCollector** app (from your LinkSquare NIR device)

---

**Step-by-step guide:**

**① Select a medicine**
Browse the list below and click **"View Medicine Info"** on the medicine you want to compare against.

**② View the reference spectrum**
You'll see the NIR spectrum of the reference medicine — this is what a genuine sample looks like.

**③ Upload your CSV**
Click **"Upload CSV"** and select the CSV file exported from LSCollector for your sample.

**④ Read the results**
The app will automatically analyse your sample and show:
- 📊 A table of polynomial fit results
- 🔬 Self Accuracy — how well the model fits the reference
- 🧪 Test Accuracy — how well the same model fits your sample
- 🎯 **Similarity %** — the key result:
  - **~100%** → your sample closely matches the reference medicine ✅
  - **Much lower** → the spectra differ significantly ⚠️

**⑤ Download the PDF report**
A PDF report is generated automatically — click **"Download PDF Report"** to save it.

---

💡 **Tip:** You can switch between English and 繁體中文 using the button in the top-right corner.
""")
    else:
        with st.expander("ℹ️ 初次使用？了解如何使用本應用程式", expanded=False):
            st.markdown("""
**歡迎使用 Filix Medtech – 近紅外光譜檢視器** 👋

本應用程式利用近紅外光譜（NIR）技術，協助您驗證藥品樣本是否與已知參考藥品相符。

---

**開始前您需要準備：**
- 從 **LSCollector** 應用程式（配合 LinkSquare NIR 裝置）匯出的 CSV 檔案

---

**使用步驟：**

**① 選擇藥品**
瀏覽下方列表，點擊您想比較的藥品旁的 **「查看藥品資訊」**。

**② 查看參考光譜**
您將看到參考藥品的近紅外光譜——這是正品樣本的光譜形狀。

**③ 上傳您的 CSV**
點擊 **「上傳 CSV」**，選擇從 LSCollector 匯出的樣本 CSV 檔案。

**④ 閱讀分析結果**
應用程式將自動分析您的樣本並顯示：
- 📊 多項式擬合結果表格
- 🔬 自身準確度——模型對參考藥品的擬合程度
- 🧪 測試準確度——相同模型對您樣本的擬合程度
- 🎯 **相似度 %**——最關鍵的結果：
  - **約 100%** → 您的樣本與參考藥品高度相符 ✅
  - **遠低於 100%** → 光譜差異顯著 ⚠️

**⑤ 下載 PDF 報告**
報告將自動生成——點擊 **「下載 PDF 報告」** 即可儲存。

---

💡 **提示：** 您可以點擊右上角的按鈕在英文和繁體中文之間切換。
""")

    st.markdown("---")
    for name, info in MEDICINES.items():
        desc = info["description_zh"] if st.session_state.lang=="zh" else info["description"]
        st.markdown(f'<div class="med-card"><h3>{name}</h3><p>{desc}</p></div>',
                    unsafe_allow_html=True)
        if st.button(S["view_spectrum"], key=f"view_{name}"):
            st.session_state.page = "detail"
            st.session_state.selected_med = name
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()

# ─────────────────────────────────────────────
#  DETAIL PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == "detail":
    name = st.session_state.selected_med
    col_back, col_ttl = st.columns([0.18, 0.82])
    with col_back:
        if st.button(S["back_list"], key="back_list_btn"):
            st.session_state.page = "list"
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()
    with col_ttl:
        st.markdown(f"## {name}")

    dk = "description_zh" if st.session_state.lang=="zh" else "description"
    st.markdown(f"<p style='font-size:1.1rem;color:#555;margin-top:-8px'>{MEDICINES[name][dk]}</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"### {S['ref_spectrum']}")
    try:
        rb = _get_csv_bytes(MEDICINES[name])
        px, iv = load_spectrum(rb)
        fig = make_fig(px, iv, f"{name} — Pixel vs. Intensity", "purple")
        st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.error(f"Could not load reference spectrum: {e}")

    st.markdown("---")
    st.markdown(f"### {S['upload_section']}")
    st.markdown(f'<div class="info-note">{S["upload_note"]}</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(S["upload_btn"], type=["csv"], key="uploader")
    if uploaded is not None:
        fb = uploaded.read()
        if fb != st.session_state.upload_bytes:
            st.session_state.upload_bytes = fb
            st.session_state.analysis_res = None

    if st.session_state.upload_bytes is not None:
        if st.button(S["clear_btn"], key="clear_btn"):
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()
        try:
            upx, uiv = load_spectrum(st.session_state.upload_bytes)
            ufig = make_fig(upx, uiv,
                            f"{uploaded.name if uploaded else 'Sample'} — Pixel vs. Intensity",
                            "darkorange")
            st.pyplot(ufig); plt.close(ufig)
        except Exception as e:
            st.error(f"Could not load sample spectrum: {e}")

        if st.session_state.analysis_res is None:
            with st.spinner(S["running"]):
                try:
                    sb = _get_csv_bytes(MEDICINES[name])
                    st.session_state.analysis_res = run_analysis(sb, st.session_state.upload_bytes)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if st.session_state.analysis_res:
            res = st.session_state.analysis_res
            st.markdown("---")
            st.markdown(f"### {S['analysis_title']}")
            df_t = pd.DataFrame(res["table"])
            df_t.rename(columns={"degree":S["degree"],"Adj. R²":S["adj_r2"],
                                  "R²":S["r2"],"Adj.R²/R² (%)":S["ratio"]}, inplace=True)
            st.dataframe(df_t, use_container_width=True, hide_index=True)
            c1,c2,c3 = st.columns(3)
            c1.metric(S["chosen_degree"], res["best_degree"])
            c2.metric(S["self_acc"], f"{res['self_acc']:.6f}")
            c3.metric(S["test_acc"], f"{res['test_acc']:.6f}")
            c4,c5 = st.columns(2)
            c4.metric(S["best_ratio"], f"{res['best_ratio']:.4f} %")
            sim = res["similarity"]
            c5.metric(S["similarity_result"], f"{sim:.4f} %")
            st.markdown(f'<div class="similarity-box">{S["similarity_result"]}: {sim:.2f}%</div>',
                        unsafe_allow_html=True)
            with st.expander(S["how_to_read"]):
                for t,d in S["explanations"]:
                    st.markdown(f"**{t}**"); st.markdown(d); st.markdown("---")

            st.markdown("---")
            with st.spinner("📄 Preparing PDF report…"):
                try:
                    rb2 = _get_csv_bytes(MEDICINES[name])
                    px2,iv2 = load_spectrum(rb2)
                    fr2 = make_fig(px2,iv2,f"{name} — Pixel vs. Intensity","purple")
                    ref_b = fig_to_bytes(fr2); plt.close(fr2)
                    upx2,uiv2 = load_spectrum(st.session_state.upload_bytes)
                    sl = uploaded.name if uploaded else "Sample"
                    fu2 = make_fig(upx2,uiv2,f"{sl} — Pixel vs. Intensity","darkorange")
                    smp_b = fig_to_bytes(fu2); plt.close(fu2)
                    pdf_b = generate_pdf(name, sl, ref_b, smp_b, res, st.session_state.lang)
                    fname = f"filix_report_{name.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button("📥 Download PDF Report", pdf_b, fname, "application/pdf", key="pdf_dl")
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")
