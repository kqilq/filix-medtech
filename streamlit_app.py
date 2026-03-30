"""
Filix Medtech – NIR Spectrum Viewer  (Streamlit web version)
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
import os
import io

# ─────────────────────────────────────────────
#  Paths & registry
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "filix_logo.png")

MEDICINES = {
    "Medicine 1": {
        "csv": os.path.join(BASE_DIR, "medicine1.csv"),
        "description": "NIR spectrum captured with LinkSquare device. Wavelength range: ~704 – 1088 nm",
        "description_zh": "以 LinkSquare 裝置擷取的近紅外光譜。波長範圍：約 704 – 1088 nm",
    },
    "Medicine 2": {
        "csv": os.path.join(BASE_DIR, "medicine2.csv"),
        "description": "NIR spectrum captured with LinkSquare device. Wavelength range: ~704 – 1088 nm",
        "description_zh": "以 LinkSquare 裝置擷取的近紅外光譜。波長範圍：約 704 – 1088 nm",
    },
}

# ─────────────────────────────────────────────
#  Localisation
# ─────────────────────────────────────────────
STRINGS = {
    "en": {
        "app_title":         "Filix Medtech – NIR Spectrum Viewer",
        "list_title":        "List of Medicines",
        "list_subtitle":     "Select a medicine to view its spectrum and compare with your sample.",
        "view_spectrum":     "View Medicine Info",
        "back_list":         "← Back to List",
        "ref_spectrum":      "Reference Spectrum",
        "upload_section":    "Compare with your CSV",
        "upload_note":       "ℹ️ Please upload a CSV file exported from LSCollector.",
        "upload_btn":        "Upload CSV",
        "clear_btn":         "Clear uploaded file",
        "analysis_title":    "Analysis Results",
        "degree":            "Degree",
        "adj_r2":            "Adj. R²",
        "r2":                "R²",
        "ratio":             "Adj.R²/R² (%)",
        "chosen_degree":     "Chosen best degree",
        "best_ratio":        "Best degree ratio to 100%",
        "self_acc":          "Self accuracy (R²)",
        "test_acc":          "Test accuracy (R²)",
        "similarity":        "Similarity",
        "similarity_result": "🎯 Similarity",
        "how_to_read":       "ℹ️ How to read these results",
        "running":           "⏳ Running analysis…",
        "language_btn":      "繁體中文",
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
        "app_title":         "Filix Medtech – 近紅外光譜檢視器",
        "list_title":        "藥品列表",
        "list_subtitle":     "選擇藥品以查看其光譜並與您的樣本比較。",
        "view_spectrum":     "查看藥品資訊",
        "back_list":         "← 返回列表",
        "ref_spectrum":      "參考光譜",
        "upload_section":    "與您的 CSV 比較",
        "upload_note":       "ℹ️ 請上傳從 LSCollector 匯出的 CSV 檔案。",
        "upload_btn":        "上傳 CSV",
        "clear_btn":         "清除已上傳檔案",
        "analysis_title":    "分析結果",
        "degree":            "階數",
        "adj_r2":            "Adj. R²",
        "r2":                "R²",
        "ratio":             "Adj.R²/R² (%)",
        "chosen_degree":     "最佳階數",
        "best_ratio":        "最佳階數與 100% 的比率",
        "self_acc":          "自身準確度 (R²)",
        "test_acc":          "測試準確度 (R²)",
        "similarity":        "相似度",
        "similarity_result": "🎯 相似度",
        "how_to_read":       "ℹ️ 如何解讀分析結果",
        "running":           "⏳ 正在分析中…",
        "language_btn":      "English",
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
#  Data helpers
# ─────────────────────────────────────────────
def clean_file(path_or_bytes, from_bytes=False):
    if from_bytes:
        df = pd.read_csv(io.BytesIO(path_or_bytes), index_col=0)
    else:
        df = pd.read_csv(path_or_bytes, index_col=0)
    df_T = df.transpose()
    mean_intensity = df_T.mean(axis=1, skipna=True)
    cleaned = pd.DataFrame({
        "Wavelength": df_T.index.astype(float),
        "Intensity":  mean_intensity.values,
    })
    return cleaned.sort_values(by="Wavelength")

def load_spectrum(path_or_bytes, from_bytes=False):
    if from_bytes:
        df = pd.read_csv(io.BytesIO(path_or_bytes), header=0)
    else:
        df = pd.read_csv(path_or_bytes, header=0)
    intensity_row    = df.iloc[0, 1:]
    intensity_values = intensity_row.astype(float).values
    return list(range(len(intensity_values))), intensity_values

def legendre_features(x, degree):
    x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    return legendre.legvander(x_scaled, degree)

def run_analysis(standard_path, test_bytes):
    standard   = clean_file(standard_path)
    test       = clean_file(test_bytes, from_bytes=True)
    wavelength = standard["Wavelength"].values
    intensity  = standard["Intensity"].values
    valid_degrees, adj_r2s = [], []
    stop_found = False
    for degree in range(2, 11):
        XX_poly = legendre_features(wavelength, degree)[:, 1:]
        XX      = sm.add_constant(XX_poly)
        model   = sm.OLS(intensity, XX).fit()
        pvals   = model.pvalues[1:]
        if np.all(pvals < 0.001) and not stop_found:
            valid_degrees.append(degree)
            adj_r2s.append(model.rsquared_adj)
        elif not stop_found:
            stop_found = True
    if not valid_degrees:
        raise ValueError("No valid polynomial degrees found.")
    X = standard[["Wavelength"]].values
    y = standard["Intensity"].values
    r_square, ratios = [], []
    for idx, degree in enumerate(valid_degrees):
        poly_reg = PolynomialFeatures(degree=degree)
        X_poly   = poly_reg.fit_transform(X)
        result   = sm.OLS(y, X_poly).fit()
        RSS = TSS = 0
        for i in range(len(y)):
            y_est = result.predict(poly_reg.fit_transform([[X[i][0]]]))[0]
            RSS  += (y[i] - y_est) ** 2
            TSS  += (y[i] - np.mean(y)) ** 2
        R_sq = 1 - RSS / TSS
        r_square.append(R_sq)
        ratios.append((adj_r2s[idx] / R_sq) * 100)
    best_idx    = int(np.argmin(np.abs(np.array(ratios) - 100)))
    best_degree = valid_degrees[best_idx]
    poly_reg    = PolynomialFeatures(degree=best_degree)
    X_poly      = poly_reg.fit_transform(X)
    final_model = sm.OLS(y, X_poly).fit()
    RSS = TSS = 0
    for i in range(len(y)):
        y_est = final_model.predict(poly_reg.fit_transform([[X[i][0]]]))[0]
        RSS  += (y[i] - y_est) ** 2
        TSS  += (y[i] - np.mean(y)) ** 2
    final_R_sq = 1 - RSS / TSS
    y_test   = test["Intensity"].values
    poly_reg = PolynomialFeatures(degree=best_degree)
    X_poly   = poly_reg.fit_transform(X)
    result   = sm.OLS(y, X_poly).fit()
    RSS = TSS = 0
    for i in range(len(y)):
        y_est  = result.predict(poly_reg.fit_transform([[X[i][0]]]))[0]
        RSS   += (y_test[i] - y_est) ** 2
        TSS   += (y_test[i] - np.mean(y)) ** 2
    R_sq_test  = 1 - RSS / TSS
    similarity = (R_sq_test / final_R_sq) * 100
    degree_table = [
        {"degree": int(d), "Adj. R²": round(a, 6),
         "R²": round(r, 6), "Adj.R²/R² (%)": round(rt, 4)}
        for d, a, r, rt in zip(valid_degrees, adj_r2s, r_square, ratios)
    ]
    return {
        "table":       degree_table,
        "best_degree": best_degree,
        "best_ratio":  ratios[best_idx],
        "self_acc":    final_R_sq,
        "test_acc":    R_sq_test,
        "similarity":  similarity,
    }

def make_spectrum_fig(px, intensity, title, color):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(px, intensity, color=color, linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Pixel Number", fontsize=11, labelpad=6)
    ax.set_ylabel("Intensity (counts)", fontsize=11, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Filix Medtech – NIR Spectrum Viewer",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px !important; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.7rem !important; }
    h3 { font-size: 1.4rem !important; }
    p, label, .stMarkdown, .stCaption,
    .stText, div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 0.6rem 1.4rem !important;
    }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .stDataFrame { font-size: 1rem !important; }
    .med-card {
        background: white; border: 1px solid #ddd;
        border-radius: 10px; padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .med-card h3 { margin: 0 0 6px 0; color: #3a3a5c; font-size: 1.3rem; }
    .med-card p  { margin: 0 0 12px 0; color: #555; font-size: 1.05rem; }
    .info-note {
        background: #eef4fb; border: 1px solid #b0cce8;
        border-radius: 6px; padding: 10px 16px;
        color: #2a5080; font-size: 1rem; margin-bottom: 12px;
    }
    .similarity-box {
        background: #f0fff0; border: 2px solid #5a7a5c;
        border-radius: 8px; padding: 16px 24px;
        text-align: center; font-size: 1.6rem;
        font-weight: bold; color: #3a3a5c; margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "lang"         not in st.session_state: st.session_state.lang         = "en"
if "page"         not in st.session_state: st.session_state.page         = "list"
if "selected_med" not in st.session_state: st.session_state.selected_med = None
if "upload_bytes" not in st.session_state: st.session_state.upload_bytes = None
if "analysis_res" not in st.session_state: st.session_state.analysis_res = None

S = STRINGS[st.session_state.lang]

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
col_logo, col_title, col_lang = st.columns([0.08, 0.82, 0.10])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=56)
with col_title:
    st.markdown(
        f"<div style='padding-top:8px'><h2 style='color:#3a3a5c;margin:0'>"
        f"{S['app_title']}</h2></div>",
        unsafe_allow_html=True)
with col_lang:
    if st.button(S["language_btn"], key="lang_btn"):
        st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
        st.rerun()

st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIST PAGE (default landing page)
# ─────────────────────────────────────────────
if st.session_state.page == "list":
    st.markdown(f"## {S['list_title']}")
    st.markdown(f"<p style='color:#555;margin-top:-8px'>{S['list_subtitle']}</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    for name, info in MEDICINES.items():
        desc = info["description_zh"] if st.session_state.lang == "zh" else info["description"]
        st.markdown(f'<div class="med-card"><h3>{name}</h3><p>{desc}</p></div>',
                    unsafe_allow_html=True)
        if st.button(S["view_spectrum"], key=f"view_{name}"):
            st.session_state.page         = "detail"
            st.session_state.selected_med = name
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()

# ─────────────────────────────────────────────
#  DETAIL PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == "detail":
    name = st.session_state.selected_med
    col_back, col_title_d = st.columns([0.18, 0.82])
    with col_back:
        if st.button(S["back_list"], key="back_list_btn"):
            st.session_state.page         = "list"
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()
    with col_title_d:
        st.markdown(f"## {name}")

    desc_key = "description_zh" if st.session_state.lang == "zh" else "description"
    st.markdown(
        f"<p style='font-size:1.1rem;color:#555;margin-top:-8px'>"
        f"{MEDICINES[name][desc_key]}</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # Reference spectrum
    st.markdown(f"### {S['ref_spectrum']}")
    try:
        px, intensity = load_spectrum(MEDICINES[name]["csv"])
        fig_ref = make_spectrum_fig(px, intensity, f"{name} — Pixel vs. Intensity", "purple")
        st.pyplot(fig_ref); plt.close(fig_ref)
    except Exception as e:
        st.error(f"Could not load reference spectrum: {e}")

    st.markdown("---")
    st.markdown(f"### {S['upload_section']}")
    st.markdown(f'<div class="info-note">{S["upload_note"]}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(S["upload_btn"], type=["csv"], key="uploader")

    if uploaded is not None:
        file_bytes = uploaded.read()
        if file_bytes != st.session_state.upload_bytes:
            st.session_state.upload_bytes = file_bytes
            st.session_state.analysis_res = None

    if st.session_state.upload_bytes is not None:
        if st.button(S["clear_btn"], key="clear_btn"):
            st.session_state.upload_bytes = None
            st.session_state.analysis_res = None
            st.rerun()

        # Show uploaded spectrum
        try:
            u_px, u_int = load_spectrum(st.session_state.upload_bytes, from_bytes=True)
            fig_user = make_spectrum_fig(u_px, u_int,
                                         f"{uploaded.name if uploaded else 'Sample'} — Pixel vs. Intensity",
                                         "darkorange")
            st.pyplot(fig_user); plt.close(fig_user)
        except Exception as e:
            st.error(f"Could not load sample spectrum: {e}")

        # Run analysis
        if st.session_state.analysis_res is None:
            with st.spinner(S["running"]):
                try:
                    res = run_analysis(MEDICINES[name]["csv"], st.session_state.upload_bytes)
                    st.session_state.analysis_res = res
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if st.session_state.analysis_res:
            res = st.session_state.analysis_res
            st.markdown("---")
            st.markdown(f"### {S['analysis_title']}")
            df_table = pd.DataFrame(res["table"])
            df_table.rename(columns={
                "degree": S["degree"], "Adj. R²": S["adj_r2"],
                "R²": S["r2"], "Adj.R²/R² (%)": S["ratio"],
            }, inplace=True)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            col1, col2, col3 = st.columns(3)
            col1.metric(S["chosen_degree"], res["best_degree"])
            col2.metric(S["self_acc"],  f"{res['self_acc']:.6f}")
            col3.metric(S["test_acc"],  f"{res['test_acc']:.6f}")
            col4, col5 = st.columns(2)
            col4.metric(S["best_ratio"], f"{res['best_ratio']:.4f} %")
            sim = res["similarity"]
            col5.metric(S["similarity_result"], f"{sim:.4f} %")
            st.markdown(
                f'<div class="similarity-box">{S["similarity_result"]}: {sim:.2f}%</div>',
                unsafe_allow_html=True)
            with st.expander(S["how_to_read"]):
                for title, desc in S["explanations"]:
                    st.markdown(f"**{title}**")
                    st.markdown(desc)
                    st.markdown("---")
