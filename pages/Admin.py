"""
Filix Medtech – Admin Panel
Manage the medicines available for users to compare against.
Uses GitHub API to read/write medicines.json so changes persist on Streamlit Cloud.
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
#  Paths (local fallback)
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "filix_logo.png")

# ─────────────────────────────────────────────
#  GitHub API config
# ─────────────────────────────────────────────
GITHUB_REPO  = "kqilq/filix-medtech"
GITHUB_BRANCH = "main"
JSON_PATH    = "medicines.json"          # path inside the repo
DATA_FOLDER  = "medicines_data"          # folder inside the repo for CSVs

def _gh_token():
    """Get GitHub PAT from Streamlit secrets or environment."""
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return os.environ.get("GITHUB_TOKEN", "")

def _gh_headers():
    token = _gh_token()
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

def _gh_api(path):
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

# ─────────────────────────────────────────────
#  GitHub read/write helpers
# ─────────────────────────────────────────────
def gh_read_json():
    """Read medicines.json from GitHub. Returns (list_of_medicines, sha)."""
    r = requests.get(_gh_api(JSON_PATH), headers=_gh_headers(),
                     params={"ref": GITHUB_BRANCH})
    if r.status_code == 404:
        return [], None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    medicines = json.loads(content).get("medicines", [])
    return medicines, data["sha"]


def gh_write_json(medicines_list, sha, message):
    """Write medicines.json back to GitHub."""
    content = json.dumps({"medicines": medicines_list}, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload = {
        "message": message,
        "content": encoded,
        "branch":  GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(_gh_api(JSON_PATH), headers=_gh_headers(), json=payload)
    r.raise_for_status()
    return r.json()


def gh_upload_csv(filename, csv_bytes, message):
    """Upload a CSV file to medicines_data/ in the GitHub repo."""
    path = f"{DATA_FOLDER}/{filename}"
    # Check if file already exists (need sha to update)
    r = requests.get(_gh_api(path), headers=_gh_headers(),
                     params={"ref": GITHUB_BRANCH})
    sha = r.json().get("sha") if r.status_code == 200 else None
    encoded = base64.b64encode(csv_bytes).decode("utf-8")
    payload = {
        "message": message,
        "content": encoded,
        "branch":  GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(_gh_api(path), headers=_gh_headers(), json=payload)
    r.raise_for_status()


def gh_delete_file(path, sha, message):
    """Delete a file from the GitHub repo."""
    payload = {
        "message": message,
        "sha":     sha,
        "branch":  GITHUB_BRANCH,
    }
    r = requests.delete(_gh_api(path), headers=_gh_headers(), json=payload)
    r.raise_for_status()


def gh_get_file_sha(path):
    """Get the SHA of a file in the repo (needed for deletion)."""
    r = requests.get(_gh_api(path), headers=_gh_headers(),
                     params={"ref": GITHUB_BRANCH})
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def gh_read_csv(filename):
    """Read a CSV file from the GitHub repo and return bytes."""
    # Try medicines_data/ first, then root
    for path in [f"{DATA_FOLDER}/{filename}", filename]:
        r = requests.get(_gh_api(path), headers=_gh_headers(),
                         params={"ref": GITHUB_BRANCH})
        if r.status_code == 200:
            return base64.b64decode(r.json()["content"])
    return None


def safe_filename(name: str) -> str:
    """Convert a medicine name to a safe filename."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name


# ─────────────────────────────────────────────
#  Spectrum preview helper
# ─────────────────────────────────────────────
def preview_spectrum(csv_bytes, title="Spectrum Preview", color="purple"):
    df = pd.read_csv(io.BytesIO(csv_bytes), header=0)
    intensity_row = df.iloc[0, 1:]
    intensity_values = intensity_row.astype(float).values
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
if "confirm_delete" not in st.session_state: st.session_state.confirm_delete = None
if "add_success"    not in st.session_state: st.session_state.add_success    = False
if "delete_success" not in st.session_state: st.session_state.delete_success = None

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=52)
with col_title:
    st.markdown(
        "<div style='padding-top:6px'>"
        "<h2 style='color:#3a3a5c;margin:0'>Filix Medtech – Admin Panel</h2>"
        "<p style='color:#888;margin:0;font-size:0.95rem'>Manage reference medicines for user comparison</p>"
        "</div>",
        unsafe_allow_html=True,
    )
st.markdown("<hr style='margin:8px 0 20px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Check GitHub token
# ─────────────────────────────────────────────
if not _gh_token():
    st.error(
        "⚠️ **GitHub token not configured.**\n\n"
        "Please add `GITHUB_TOKEN` to your Streamlit Cloud secrets:\n"
        "1. Go to your app settings on share.streamlit.io\n"
        "2. Click **Secrets**\n"
        "3. Add: `GITHUB_TOKEN = \"your_token_here\"`"
    )
    st.stop()

# ─────────────────────────────────────────────
#  Load medicines from GitHub
# ─────────────────────────────────────────────
try:
    medicines, json_sha = gh_read_json()
except Exception as e:
    st.error(f"❌ Could not load medicines from GitHub: {e}")
    st.stop()

# ── Success / delete banners ──────────────────
if st.session_state.add_success:
    st.markdown('<div class="success-box">✅ Medicine added successfully! The user app will update shortly.</div>',
                unsafe_allow_html=True)
    st.session_state.add_success = False

if st.session_state.delete_success:
    st.markdown(
        f'<div class="success-box">🗑️ <b>{st.session_state.delete_success}</b> has been removed. '
        f'The user app will update shortly.</div>',
        unsafe_allow_html=True,
    )
    st.session_state.delete_success = None

# ═══════════════════════════════════════════════
#  SECTION 1 – Current Medicines
# ═══════════════════════════════════════════════
st.markdown("## 📋 Current Medicines")
st.markdown(
    "<p style='color:#666;margin-top:-8px'>These medicines are currently available for users to compare against.</p>",
    unsafe_allow_html=True,
)

if not medicines:
    st.info("ℹ️ No medicines registered yet. Add one below.")
else:
    for idx, med in enumerate(medicines):
        st.markdown(
            f'<div class="med-row">'
            f'<h4>💊 {med["name"]}</h4>'
            f'<p><b>EN:</b> {med.get("description", "—")}</p>'
            f'<p><b>ZH:</b> {med.get("description_zh", "—")}</p>'
            f'<p style="font-size:0.85rem;color:#aaa;margin-top:4px">CSV: {med["csv"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_prev, col_del = st.columns([0.75, 0.25])

        with col_prev:
            with st.expander(f"📈 Preview spectrum – {med['name']}"):
                try:
                    csv_bytes = gh_read_csv(med["csv"])
                    if csv_bytes:
                        fig = preview_spectrum(csv_bytes, title=f"{med['name']} – Pixel vs. Intensity")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("CSV file not found in repository.")
                except Exception as e:
                    st.error(f"Could not preview: {e}")

        with col_del:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.session_state.confirm_delete == med["name"]:
                st.markdown(
                    '<div class="danger-box">⚠️ Are you sure? This cannot be undone.</div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, delete", key=f"confirm_yes_{idx}"):
                        try:
                            # Remove from medicines list
                            new_list = [m for m in medicines if m["name"] != med["name"]]
                            _, current_sha = gh_read_json()
                            gh_write_json(new_list, current_sha,
                                          f"Admin: remove medicine '{med['name']}'")
                            # Try to delete the CSV from medicines_data/ if it's there
                            csv_fn = med["csv"]
                            csv_path_in_repo = f"{DATA_FOLDER}/{csv_fn}"
                            csv_sha = gh_get_file_sha(csv_path_in_repo)
                            if csv_sha:
                                gh_delete_file(csv_path_in_repo, csv_sha,
                                               f"Admin: remove CSV for '{med['name']}'")
                            st.session_state.confirm_delete = None
                            st.session_state.delete_success = med["name"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete: {e}")
                with c2:
                    if st.button("❌ Cancel", key=f"confirm_no_{idx}"):
                        st.session_state.confirm_delete = None
                        st.rerun()
            else:
                if st.button("🗑️ Delete", key=f"delete_{idx}", type="secondary"):
                    st.session_state.confirm_delete = med["name"]
                    st.rerun()

        st.markdown("")

# ═══════════════════════════════════════════════
#  SECTION 2 – Add New Medicine
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## ➕ Add New Medicine")
st.markdown(
    "<p style='color:#666;margin-top:-8px'>"
    "Upload a CSV exported from LSCollector and fill in the medicine details.</p>",
    unsafe_allow_html=True,
)

with st.form("add_medicine_form", clear_on_submit=True):
    st.markdown("### Medicine Details")
    col_a, col_b = st.columns(2)
    with col_a:
        med_name = st.text_input(
            "Medicine Name *",
            placeholder="e.g. Paracetamol 500mg",
            help="This name will be shown to users in the medicine list.",
        )
    with col_b:
        uploaded_csv = st.file_uploader(
            "Reference CSV (from LSCollector) *",
            type=["csv"],
            help="Upload the CSV file exported from the LinkSquare LSCollector app.",
        )
    desc_en = st.text_area(
        "Description (English) *",
        placeholder="e.g. NIR spectrum captured with LinkSquare device. Wavelength range: ~704 – 1088 nm",
        height=80,
    )
    desc_zh = st.text_area(
        "Description (繁體中文)",
        placeholder="e.g. 以 LinkSquare 裝置擷取的近紅外光譜。波長範圍：約 704 – 1088 nm",
        height=80,
    )
    submitted = st.form_submit_button("💾 Save Medicine", use_container_width=True, type="primary")

    if submitted:
        errors = []
        if not med_name.strip():
            errors.append("Medicine name is required.")
        if uploaded_csv is None:
            errors.append("Please upload a CSV file.")
        if not desc_en.strip():
            errors.append("English description is required.")
        existing_names = [m["name"].strip().lower() for m in medicines]
        if med_name.strip().lower() in existing_names:
            errors.append(f'A medicine named "{med_name.strip()}" already exists.')

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            csv_bytes = uploaded_csv.read()
            # Validate CSV
            try:
                df_check = pd.read_csv(io.BytesIO(csv_bytes), header=0)
                if df_check.shape[0] < 1 or df_check.shape[1] < 2:
                    st.error("❌ The CSV file appears to be empty or has too few columns.")
                    st.stop()
                _ = df_check.iloc[0, 1:].astype(float).values
            except Exception as e:
                st.error(f"❌ Could not parse the CSV file: {e}")
                st.stop()

            # Build safe filename
            safe_name = safe_filename(med_name.strip())
            csv_filename = f"{safe_name}.csv"

            try:
                with st.spinner("⏳ Saving to GitHub…"):
                    # Upload CSV to medicines_data/
                    gh_upload_csv(
                        csv_filename, csv_bytes,
                        f"Admin: add CSV for '{med_name.strip()}'"
                    )
                    # Update medicines.json
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
                st.error(f"❌ Failed to save to GitHub: {e}")

# ── Live CSV preview ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Preview a CSV before adding")
st.markdown(
    "<p style='color:#666;margin-top:-8px'>"
    "Upload a CSV here to preview its spectrum before saving it as a medicine.</p>",
    unsafe_allow_html=True,
)
preview_file = st.file_uploader("Upload CSV to preview", type=["csv"], key="preview_uploader")
if preview_file is not None:
    try:
        raw = preview_file.read()
        fig = preview_spectrum(raw, title=f"{preview_file.name} – Spectrum Preview", color="darkorange")
        st.pyplot(fig)
        plt.close(fig)
        df_prev = pd.read_csv(io.BytesIO(raw), header=0)
        st.caption(
            f"✅ Valid CSV — {df_prev.shape[0]} scan row(s), "
            f"{df_prev.shape[1] - 1} wavelength points"
        )
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#aaa;font-size:0.85rem;text-align:center'>"
    "Filix Medtech Admin Panel · Changes are committed to GitHub and reflected in the user app automatically</p>",
    unsafe_allow_html=True,
)
