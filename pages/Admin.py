"""
Filix Medtech – Admin Panel (bilingual EN / 繁體中文)
Admin page: https://filix-medtech-heaudbkmpj29m6ytihv8bg.streamlit.app/Admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import io
import json
import re
import base64
import requests

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
LOGO_PATH = os.path.join(BASE_DIR, "filix_logo.png")

# ─────────────────────────────────────────────
#  GitHub API config
# ─────────────────────────────────────────────
GITHUB_REPO   = "kqilq/filix-medtech"
GITHUB_BRANCH = "main"
JSON_PATH     = "medicines.json"
DATA_FOLDER   = "medicines_data"

def _gh_token():
    try:    return st.secrets["GITHUB_TOKEN"]
    except: return os.environ.get("GITHUB_TOKEN", "")

def _gh_headers():
    token = _gh_token()
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

def _gh_api(path):
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

def gh_read_json():
    r = requests.get(_gh_api(JSON_PATH), headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
    if r.status_code == 404: return [], None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return json.loads(content).get("medicines", []), data["sha"]

def gh_write_json(medicines_list, sha, message):
    content = json.dumps({"medicines": medicines_list}, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload = {"message": message, "content": encoded, "branch": GITHUB_BRANCH}
    if sha: payload["sha"] = sha
    r = requests.put(_gh_api(JSON_PATH), headers=_gh_headers(), json=payload)
    r.raise_for_status()

def gh_upload_csv(filename, csv_bytes, message):
    path = f"{DATA_FOLDER}/{filename}"
    r = requests.get(_gh_api(path), headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
    sha = r.json().get("sha") if r.status_code == 200 else None
    encoded = base64.b64encode(csv_bytes).decode("utf-8")
    payload = {"message": message, "content": encoded, "branch": GITHUB_BRANCH}
    if sha: payload["sha"] = sha
    r = requests.put(_gh_api(path), headers=_gh_headers(), json=payload)
    r.raise_for_status()

def gh_delete_file(path, sha, message):
    r = requests.delete(_gh_api(path), headers=_gh_headers(),
                        json={"message": message, "sha": sha, "branch": GITHUB_BRANCH})
    r.raise_for_status()

def gh_get_file_sha(path):
    r = requests.get(_gh_api(path), headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
    return r.json().get("sha") if r.status_code == 200 else None

def gh_read_csv(filename):
    for path in [f"{DATA_FOLDER}/{filename}", filename]:
        r = requests.get(_gh_api(path), headers=_gh_headers(), params={"ref": GITHUB_BRANCH})
        if r.status_code == 200:
            return base64.b64decode(r.json()["content"])
    return None

def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    return re.sub(r"[\s]+", "_", name)

# ─────────────────────────────────────────────
#  Localisation
# ─────────────────────────────────────────────
STRINGS = {
    "en": {
        "lang_btn":        "繁體中文",
        "subtitle":        "Manage reference medicines for user comparison",
        "token_error":     "⚠️ **GitHub token not configured.**\n\nPlease add `GITHUB_TOKEN` to your Streamlit Cloud secrets.",
        "load_error":      "❌ Could not load medicines from GitHub: ",
        "add_success":     "✅ Medicine added successfully! The user app will update shortly.",
        "del_success":     "🗑️ **{name}** has been removed. The user app will update shortly.",
        "current_title":   "## 📋 Current Medicines",
        "current_sub":     "These medicines are currently available for users to compare against.",
        "no_medicines":    "ℹ️ No medicines registered yet. Add one below.",
        "label_en":        "EN",
        "label_zh":        "ZH",
        "label_csv":       "CSV",
        "preview_btn":     "📈 Preview spectrum – ",
        "csv_not_found":   "CSV file not found in repository.",
        "preview_error":   "Could not preview: ",
        "confirm_msg":     "⚠️ Are you sure? This cannot be undone.",
        "yes_delete":      "✅ Yes, delete",
        "cancel":          "❌ Cancel",
        "delete_fail":     "❌ Failed to delete: ",
        "delete_btn":      "🗑️ Delete",
        "add_title":       "## ➕ Add New Medicine",
        "add_sub":         "Upload a CSV exported from LSCollector and fill in the medicine details.",
        "form_title":      "### Medicine Details",
        "name_label":      "Medicine Name *",
        "name_ph":         "e.g. Paracetamol 500mg",
        "name_help":       "This name will be shown to users in the medicine list.",
        "csv_label":       "Reference CSV (from LSCollector) *",
        "csv_help":        "Upload the CSV file exported from the LinkSquare LSCollector app.",
        "desc_en_label":   "Description (English) *",
        "desc_en_ph":      "e.g. NIR spectrum captured with LinkSquare device. Wavelength range: ~704 – 1088 nm",
        "desc_zh_label":   "Description (繁體中文)",
        "desc_zh_ph":      "e.g. 以 LinkSquare 裝置擷取的近紅外光譜。波長範圍：約 704 – 1088 nm",
        "save_btn":        "💾 Save Medicine",
        "err_name":        "Medicine name is required.",
        "err_csv":         "Please upload a CSV file.",
        "err_desc":        "English description is required.",
        "err_exists":      'A medicine named "{name}" already exists.',
        "err_csv_empty":   "❌ The CSV file appears to be empty or has too few columns.",
        "err_csv_parse":   "❌ Could not parse the CSV file: ",
        "saving":          "⏳ Saving to GitHub…",
        "save_fail":       "❌ Failed to save to GitHub: ",
        "preview_title":   "### 🔍 Preview a CSV before adding",
        "preview_sub":     "Upload a CSV here to preview its spectrum before saving it as a medicine.",
        "preview_upload":  "Upload CSV to preview",
        "preview_valid":   "✅ Valid CSV — {rows} scan row(s), {cols} wavelength points",
        "preview_err":     "❌ Could not read CSV: ",
        "footer":          "Filix Medtech Admin Panel · Changes are committed to GitHub and reflected in the user app automatically",
        "spectrum_title":  " – Pixel vs. Intensity",
    },
    "zh": {
        "lang_btn":        "English",
        "subtitle":        "管理供用戶比較的參考藥品",
        "token_error":     "⚠️ **GitHub 令牌未設定。**\n\n請在 Streamlit Cloud 的 Secrets 中加入 `GITHUB_TOKEN`。",
        "load_error":      "❌ 無法從 GitHub 載入藥品資料：",
        "add_success":     "✅ 藥品已成功新增！用戶頁面將很快更新。",
        "del_success":     "🗑️ **{name}** 已被移除。用戶頁面將很快更新。",
        "current_title":   "## 📋 目前藥品列表",
        "current_sub":     "以下藥品目前可供用戶進行比較。",
        "no_medicines":    "ℹ️ 尚未登記任何藥品。請在下方新增。",
        "label_en":        "英文",
        "label_zh":        "中文",
        "label_csv":       "CSV",
        "preview_btn":     "📈 預覽光譜 – ",
        "csv_not_found":   "在儲存庫中找不到 CSV 檔案。",
        "preview_error":   "無法預覽：",
        "confirm_msg":     "⚠️ 確定要刪除嗎？此操作無法復原。",
        "yes_delete":      "✅ 確認刪除",
        "cancel":          "❌ 取消",
        "delete_fail":     "❌ 刪除失敗：",
        "delete_btn":      "🗑️ 刪除",
        "add_title":       "## ➕ 新增藥品",
        "add_sub":         "上傳從 LSCollector 匯出的 CSV 並填寫藥品資料。",
        "form_title":      "### 藥品資料",
        "name_label":      "藥品名稱 *",
        "name_ph":         "例如：Paracetamol 500mg",
        "name_help":       "此名稱將顯示在用戶的藥品列表中。",
        "csv_label":       "參考 CSV（來自 LSCollector）*",
        "csv_help":        "上傳從 LinkSquare LSCollector 應用程式匯出的 CSV 檔案。",
        "desc_en_label":   "英文描述 *",
        "desc_en_ph":      "例如：NIR spectrum captured with LinkSquare device. Wavelength range: ~704 – 1088 nm",
        "desc_zh_label":   "繁體中文描述",
        "desc_zh_ph":      "例如：以 LinkSquare 裝置擷取的近紅外光譜。波長範圍：約 704 – 1088 nm",
        "save_btn":        "💾 儲存藥品",
        "err_name":        "藥品名稱為必填項目。",
        "err_csv":         "請上傳 CSV 檔案。",
        "err_desc":        "英文描述為必填項目。",
        "err_exists":      '名為「{name}」的藥品已存在。',
        "err_csv_empty":   "❌ CSV 檔案似乎為空或欄位太少。",
        "err_csv_parse":   "❌ 無法解析 CSV 檔案：",
        "saving":          "⏳ 正在儲存至 GitHub…",
        "save_fail":       "❌ 儲存至 GitHub 失敗：",
        "preview_title":   "### 🔍 新增前預覽 CSV",
        "preview_sub":     "在此上傳 CSV 以預覽其光譜，然後再儲存為藥品。",
        "preview_upload":  "上傳 CSV 以預覽",
        "preview_valid":   "✅ 有效 CSV — {rows} 行掃描資料，{cols} 個波長點",
        "preview_err":     "❌ 無法讀取 CSV：",
        "footer":          "Filix Medtech 管理面板 · 變更已提交至 GitHub，並自動反映在用戶頁面",
        "spectrum_title":  " – 像素 vs. 強度",
    },
}

# ─────────────────────────────────────────────
#  Spectrum preview helper
# ─────────────────────────────────────────────
def preview_spectrum(csv_bytes, title="Spectrum Preview", color="purple"):
    df = pd.read_csv(io.BytesIO(csv_bytes), header=0)
    intensity_values = df.iloc[0, 1:].astype(float).values
    px = list(range(len(intensity_values)))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(px, intensity_values, color=color, linewidth=1.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel Number", fontsize=10)
    ax.set_ylabel("Intensity (counts)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Filix Medtech – Admin Panel",
    page_icon="🛠️",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    html, body, [class*="css"] { font-size: 16px !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }
    .stButton > button { font-size: 1rem !important; padding: 0.45rem 1.1rem !important; }
    .med-row {
        background: #fff; border: 1px solid #ddd; border-radius: 8px;
        padding: 14px 20px; margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .med-row h4 { margin: 0 0 4px 0; color: #3a3a5c; font-size: 1.1rem; }
    .med-row p  { margin: 0; color: #666; font-size: 0.95rem; }
    .success-box {
        background: #f0fff0; border: 1px solid #5a7a5c;
        border-radius: 6px; padding: 10px 16px; color: #2d5a2d; margin: 8px 0;
    }
    .danger-box {
        background: #fff0f0; border: 1px solid #c0392b;
        border-radius: 6px; padding: 10px 16px; color: #7a1010; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
for k, v in [("admin_lang", "en"), ("confirm_delete", None),
              ("add_success", False), ("delete_success", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

S = STRINGS[st.session_state.admin_lang]

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
col_logo, col_title, col_lang = st.columns([0.08, 0.82, 0.10])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=52)
with col_title:
    st.markdown(
        f"<div style='padding-top:6px'>"
        f"<h2 style='color:#3a3a5c;margin:0'>Filix Medtech – "
        f"{'管理面板' if st.session_state.admin_lang == 'zh' else 'Admin Panel'}</h2>"
        f"<p style='color:#888;margin:0;font-size:0.95rem'>{S['subtitle']}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_lang:
    if st.button(S["lang_btn"], key="admin_lang_btn"):
        st.session_state.admin_lang = "zh" if st.session_state.admin_lang == "en" else "en"
        st.rerun()

st.markdown("<hr style='margin:8px 0 20px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Check GitHub token
# ─────────────────────────────────────────────
if not _gh_token():
    st.error(S["token_error"])
    st.stop()

# ─────────────────────────────────────────────
#  Load medicines from GitHub
# ─────────────────────────────────────────────
try:
    medicines, json_sha = gh_read_json()
except Exception as e:
    st.error(f"{S['load_error']}{e}")
    st.stop()

# ── Success / delete banners ──────────────────
if st.session_state.add_success:
    st.markdown(f'<div class="success-box">{S["add_success"]}</div>', unsafe_allow_html=True)
    st.session_state.add_success = False

if st.session_state.delete_success:
    msg = S["del_success"].format(name=st.session_state.delete_success)
    st.markdown(f'<div class="success-box">{msg}</div>', unsafe_allow_html=True)
    st.session_state.delete_success = None

# ═══════════════════════════════════════════════
#  SECTION 1 – Current Medicines
# ═══════════════════════════════════════════════
st.markdown(S["current_title"])
st.markdown(f"<p style='color:#666;margin-top:-8px'>{S['current_sub']}</p>",
            unsafe_allow_html=True)

if not medicines:
    st.info(S["no_medicines"])
else:
    for idx, med in enumerate(medicines):
        st.markdown(
            f'<div class="med-row">'
            f'<h4>💊 {med["name"]}</h4>'
            f'<p><b>{S["label_en"]}:</b> {med.get("description", "—")}</p>'
            f'<p><b>{S["label_zh"]}:</b> {med.get("description_zh", "—")}</p>'
            f'<p style="font-size:0.85rem;color:#aaa;margin-top:4px">{S["label_csv"]}: {med["csv"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_prev, col_del = st.columns([0.75, 0.25])

        with col_prev:
            with st.expander(f"{S['preview_btn']}{med['name']}"):
                try:
                    csv_bytes = gh_read_csv(med["csv"])
                    if csv_bytes:
                        fig = preview_spectrum(
                            csv_bytes,
                            title=f"{med['name']}{S['spectrum_title']}"
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning(S["csv_not_found"])
                except Exception as e:
                    st.error(f"{S['preview_error']}{e}")

        with col_del:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.session_state.confirm_delete == med["name"]:
                st.markdown(f'<div class="danger-box">{S["confirm_msg"]}</div>',
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(S["yes_delete"], key=f"confirm_yes_{idx}"):
                        try:
                            new_list = [m for m in medicines if m["name"] != med["name"]]
                            _, current_sha = gh_read_json()
                            gh_write_json(new_list, current_sha,
                                          f"Admin: remove medicine '{med['name']}'")
                            csv_path_in_repo = f"{DATA_FOLDER}/{med['csv']}"
                            csv_sha = gh_get_file_sha(csv_path_in_repo)
                            if csv_sha:
                                gh_delete_file(csv_path_in_repo, csv_sha,
                                               f"Admin: remove CSV for '{med['name']}'")
                            st.session_state.confirm_delete = None
                            st.session_state.delete_success = med["name"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"{S['delete_fail']}{e}")
                with c2:
                    if st.button(S["cancel"], key=f"confirm_no_{idx}"):
                        st.session_state.confirm_delete = None
                        st.rerun()
            else:
                if st.button(S["delete_btn"], key=f"delete_{idx}", type="secondary"):
                    st.session_state.confirm_delete = med["name"]
                    st.rerun()

        st.markdown("")

# ═══════════════════════════════════════════════
#  SECTION 2 – Add New Medicine
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown(S["add_title"])
st.markdown(f"<p style='color:#666;margin-top:-8px'>{S['add_sub']}</p>",
            unsafe_allow_html=True)

with st.form("add_medicine_form", clear_on_submit=True):
    st.markdown(S["form_title"])
    col_a, col_b = st.columns(2)
    with col_a:
        med_name = st.text_input(S["name_label"], placeholder=S["name_ph"], help=S["name_help"])
    with col_b:
        uploaded_csv = st.file_uploader(S["csv_label"], type=["csv"], help=S["csv_help"])
    desc_en = st.text_area(S["desc_en_label"], placeholder=S["desc_en_ph"], height=80)
    desc_zh = st.text_area(S["desc_zh_label"], placeholder=S["desc_zh_ph"], height=80)
    submitted = st.form_submit_button(S["save_btn"], use_container_width=True, type="primary")

    if submitted:
        errors = []
        if not med_name.strip():
            errors.append(S["err_name"])
        if uploaded_csv is None:
            errors.append(S["err_csv"])
        if not desc_en.strip():
            errors.append(S["err_desc"])
        existing_names = [m["name"].strip().lower() for m in medicines]
        if med_name.strip().lower() in existing_names:
            errors.append(S["err_exists"].format(name=med_name.strip()))

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            csv_bytes = uploaded_csv.read()
            try:
                df_check = pd.read_csv(io.BytesIO(csv_bytes), header=0)
                if df_check.shape[0] < 1 or df_check.shape[1] < 2:
                    st.error(S["err_csv_empty"]); st.stop()
                _ = df_check.iloc[0, 1:].astype(float).values
            except Exception as e:
                st.error(f"{S['err_csv_parse']}{e}"); st.stop()

            csv_filename = f"{safe_filename(med_name.strip())}.csv"
            try:
                with st.spinner(S["saving"]):
                    gh_upload_csv(csv_filename, csv_bytes,
                                  f"Admin: add CSV for '{med_name.strip()}'")
                    current_meds, current_sha = gh_read_json()
                    current_meds.append({
                        "name":           med_name.strip(),
                        "csv":            csv_filename,
                        "description":    desc_en.strip(),
                        "description_zh": desc_zh.strip(),
                    })
                    gh_write_json(current_meds, current_sha,
                                  f"Admin: add medicine '{med_name.strip()}'")
                st.session_state.add_success = True
                st.rerun()
            except Exception as e:
                st.error(f"{S['save_fail']}{e}")

# ── Live CSV preview ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(S["preview_title"])
st.markdown(f"<p style='color:#666;margin-top:-8px'>{S['preview_sub']}</p>",
            unsafe_allow_html=True)
preview_file = st.file_uploader(S["preview_upload"], type=["csv"], key="preview_uploader")
if preview_file is not None:
    try:
        raw = preview_file.read()
        fig = preview_spectrum(raw,
                               title=f"{preview_file.name}{S['spectrum_title']}",
                               color="darkorange")
        st.pyplot(fig)
        plt.close(fig)
        df_prev = pd.read_csv(io.BytesIO(raw), header=0)
        st.caption(S["preview_valid"].format(
            rows=df_prev.shape[0], cols=df_prev.shape[1]-1))
    except Exception as e:
        st.error(f"{S['preview_err']}{e}")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='color:#aaa;font-size:0.85rem;text-align:center'>{S['footer']}</p>",
    unsafe_allow_html=True,
)
