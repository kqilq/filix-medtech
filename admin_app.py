"""
Filix Medtech – Admin Panel
Manage the medicines available for users to compare against.
Changes are saved to medicines.json and persist across app restarts.
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
import shutil
import re

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH      = os.path.join(BASE_DIR, "filix_logo.png")
MEDICINES_JSON = os.path.join(BASE_DIR, "medicines.json")
MEDICINES_DATA = os.path.join(BASE_DIR, "medicines_data")

# Ensure medicines_data folder exists
os.makedirs(MEDICINES_DATA, exist_ok=True)

# ─────────────────────────────────────────────
#  JSON helpers
# ─────────────────────────────────────────────
def load_json():
    """Load medicines list from medicines.json."""
    if not os.path.exists(MEDICINES_JSON):
        return []
    with open(MEDICINES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("medicines", [])


def save_json(medicines_list):
    """Save medicines list back to medicines.json."""
    with open(MEDICINES_JSON, "w", encoding="utf-8") as f:
        json.dump({"medicines": medicines_list}, f, ensure_ascii=False, indent=2)


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
    """Return a matplotlib figure previewing the first row of a LinkSquare CSV."""
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
    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.45rem 1.1rem !important;
    }
    .med-row {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .med-row h4 { margin: 0 0 4px 0; color: #3a3a5c; font-size: 1.1rem; }
    .med-row p  { margin: 0; color: #666; font-size: 0.95rem; }
    .admin-header {
        background: linear-gradient(90deg, #3a3a5c, #5a5a8c);
        color: white; padding: 14px 24px; border-radius: 8px;
        margin-bottom: 20px;
    }
    .success-box {
        background: #f0fff0; border: 1px solid #5a7a5c;
        border-radius: 6px; padding: 10px 16px; color: #2d5a2d;
        margin: 8px 0;
    }
    .warning-box {
        background: #fff8e1; border: 1px solid #f0a500;
        border-radius: 6px; padding: 10px 16px; color: #7a5000;
        margin: 8px 0;
    }
    .danger-box {
        background: #fff0f0; border: 1px solid #c0392b;
        border-radius: 6px; padding: 10px 16px; color: #7a1010;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "confirm_delete"    not in st.session_state: st.session_state.confirm_delete    = None
if "add_success"       not in st.session_state: st.session_state.add_success       = False
if "delete_success"    not in st.session_state: st.session_state.delete_success    = None

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

# ── Success / delete banners ──────────────────
if st.session_state.add_success:
    st.markdown('<div class="success-box">✅ Medicine added successfully!</div>', unsafe_allow_html=True)
    st.session_state.add_success = False

if st.session_state.delete_success:
    st.markdown(
        f'<div class="success-box">🗑️ <b>{st.session_state.delete_success}</b> has been removed.</div>',
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

medicines = load_json()

if not medicines:
    st.info("ℹ️ No medicines registered yet. Add one below.")
else:
    for idx, med in enumerate(medicines):
        with st.container():
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
                # Resolve CSV path for preview
                csv_fn = med["csv"]
                if os.path.isabs(csv_fn):
                    csv_path = csv_fn
                elif os.path.exists(os.path.join(BASE_DIR, csv_fn)):
                    csv_path = os.path.join(BASE_DIR, csv_fn)
                elif os.path.exists(os.path.join(MEDICINES_DATA, csv_fn)):
                    csv_path = os.path.join(MEDICINES_DATA, csv_fn)
                else:
                    csv_path = None

                if csv_path and os.path.exists(csv_path):
                    with st.expander(f"📈 Preview spectrum – {med['name']}"):
                        try:
                            with open(csv_path, "rb") as f:
                                raw = f.read()
                            fig = preview_spectrum(raw, title=f"{med['name']} – Pixel vs. Intensity")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Could not preview: {e}")
                else:
                    st.warning(f"⚠️ CSV file not found: {csv_fn}")

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
                            # Remove from list
                            new_list = [m for m in medicines if m["name"] != med["name"]]
                            save_json(new_list)
                            # Optionally remove the CSV if it lives in medicines_data/
                            csv_fn2 = med["csv"]
                            candidate = os.path.join(MEDICINES_DATA, csv_fn2)
                            if os.path.exists(candidate) and not os.path.isabs(csv_fn2):
                                try:
                                    os.remove(candidate)
                                except Exception:
                                    pass
                            st.session_state.confirm_delete  = None
                            st.session_state.delete_success  = med["name"]
                            st.rerun()
                    with c2:
                        if st.button("❌ Cancel", key=f"confirm_no_{idx}"):
                            st.session_state.confirm_delete = None
                            st.rerun()
                else:
                    if st.button(f"🗑️ Delete", key=f"delete_{idx}", type="secondary"):
                        st.session_state.confirm_delete = med["name"]
                        st.rerun()

        st.markdown("")  # spacing

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
        # ── Validation ──────────────────────────────────────────────────
        errors = []
        if not med_name.strip():
            errors.append("Medicine name is required.")
        if uploaded_csv is None:
            errors.append("Please upload a CSV file.")
        if not desc_en.strip():
            errors.append("English description is required.")

        # Check for duplicate name
        existing_names = [m["name"].strip().lower() for m in load_json()]
        if med_name.strip().lower() in existing_names:
            errors.append(f'A medicine named "{med_name.strip()}" already exists.')

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            # ── Validate CSV format ──────────────────────────────────────
            csv_bytes = uploaded_csv.read()
            try:
                df_check = pd.read_csv(io.BytesIO(csv_bytes), header=0)
                if df_check.shape[0] < 1 or df_check.shape[1] < 2:
                    st.error("❌ The CSV file appears to be empty or has too few columns. "
                             "Please upload a valid LSCollector export.")
                    st.stop()
                # Try parsing intensity row
                _ = df_check.iloc[0, 1:].astype(float).values
            except Exception as e:
                st.error(f"❌ Could not parse the CSV file: {e}. "
                         "Please make sure it is a valid LSCollector export.")
                st.stop()

            # ── Save CSV to medicines_data/ ──────────────────────────────
            safe_name = safe_filename(med_name.strip())
            csv_filename = f"{safe_name}.csv"
            # Avoid overwriting existing files
            counter = 1
            while os.path.exists(os.path.join(MEDICINES_DATA, csv_filename)):
                csv_filename = f"{safe_name}_{counter}.csv"
                counter += 1

            dest_path = os.path.join(MEDICINES_DATA, csv_filename)
            with open(dest_path, "wb") as f:
                f.write(csv_bytes)

            # ── Update medicines.json ────────────────────────────────────
            current = load_json()
            current.append({
                "name":           med_name.strip(),
                "csv":            csv_filename,   # relative filename; user app resolves it
                "description":    desc_en.strip(),
                "description_zh": desc_zh.strip(),
            })
            save_json(current)

            st.session_state.add_success = True
            st.rerun()

# ── Live CSV preview (outside form) ──────────────────────────────────────
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
    "Filix Medtech Admin Panel · Changes are saved immediately and reflected in the user app</p>",
    unsafe_allow_html=True,
)
