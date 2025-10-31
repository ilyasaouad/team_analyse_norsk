# app_password = "team-analyse"
# db_user = "patent"
# db_password = "innsikt"

# Import necessary libraries
import streamlit as st
import pandas as pd
from pathlib import Path
from config import Config
import os

# Set your app password here (consider using secure storage)
APP_PASSWORD = os.getenv("APP_PASSWORD", "default_password")

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(page_title="Patent Data Viewer", page_icon="📊", layout="wide")

# ---------------------------------
# Title
# ---------------------------------
st.title("📊 Patent Applicant/Inventor Data Viewer")

# Function to check password
def check_password(password):
    return password == APP_PASSWORD

# Password input
if "password_correct" not in st.session_state:
    password = st.text_input("Enter your password:", type="password")
    if st.button("Submit"):
        if check_password(password):
            st.session_state.password_correct = True
            st.success("Access granted!")
        else:
            st.session_state.password_correct = False
            st.error("🚫 Access denied! Incorrect password.")

# If password is correct, show the app
if st.session_state.get("password_correct", False):

    @st.cache_data
    def load_data(file_path: Path) -> pd.DataFrame:
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            st.error(f"❌ File not found: {file_path}")
            return pd.DataFrame()  # Return empty DataFrame

    # ---------------------------------
    # Sidebar
    # ---------------------------------
    st.sidebar.header("Data Source")

    # Find all available DataTables directories
    current_dir = Path(__file__).parent
    available_dirs = sorted(
        [p for p in current_dir.glob("DataTables_*") if p.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # Prepare options for selectbox
    dir_options = [str(p) for p in available_dirs] if available_dirs else [str(Path(Config.output_dir))]
    selected_from_dropdown = dir_options[0]

    # Dropdown for available directories
    selected_from_dropdown = st.sidebar.selectbox(
        "📁 Available DataTables directories", options=dir_options, index=0
    )

    # Manual override option
    use_custom = st.sidebar.checkbox("Use custom path", value=False)
    selected_dir = st.sidebar.text_input("Custom directory path", value=selected_from_dropdown if not use_custom else "")
    if use_custom and not selected_dir:
        st.warning("Please enter a custom directory path.")

    base_path = Path(selected_dir).expanduser()
    families_csv = base_path / "data" / "applicants_inventors" / "unique_family_ids.csv"
    details_csv = base_path / "data" / "applicants_inventors" / "applicant_inventor_details.csv"
    summary_csv = base_path / "data" / "analysis" / "counts_ratios_summary.csv"
    
    if not base_path.exists():
        st.error(f"❌ Directory not found: {base_path}")
        st.stop()

    # ---------------------------------
    # Tabs
    # ---------------------------------
    tab_family, tab_details, tab_summary = st.tabs(
        [
            "🧬 Family IDs",
            "👤 Applicant/Inventor Details",
            "📈 Counts & Ratios Summary",
        ]
    )

    # ---------------------------------
    # Tab: Family IDs
    # ---------------------------------
    with tab_family:
        st.subheader("Unique DOCDB Family IDs")
        if not families_csv.exists():
            st.info(f"No family ID CSV found at {families_csv}")
        else:
            df_family = load_data(families_csv)
            if not df_family.empty:
                st.success(f"✅ Loaded {families_csv.name}")
                st.write(f"Records: {len(df_family)}")
                st.dataframe(df_family, use_container_width=True, height=500)
                st.download_button(
                    label="📥 Download Family IDs",
                    data=df_family.to_csv(index=False).encode("
                                        "utf-8"),
                    file_name=families_csv.name,
                    mime="text/csv",
                )

    # ---------------------------------
    # Tab: Applicant / Inventor Details
    # ---------------------------------
    with tab_details:
        st.subheader("Applicant/Inventor Details")
        if not details_csv.exists():
            st.info(f"No details CSV found at {details_csv}")
        else:
            df_details = load_data(details_csv)
            if not df_details.empty:
                st.success(f"✅ Loaded {details_csv.name}")
                st.write(f"Records: {len(df_details)}")
                st.dataframe(df_details, use_container_width=True, height=500)
                st.download_button(
                    label="📥 Download Applicant/Inventor Details",
                    data=df_details.to_csv(index=False).encode("utf-8"),
                    file_name=details_csv.name,
                    mime="text/csv",
                )

    # ---------------------------------
    # Tab: Counts & Ratios Summary
    # ---------------------------------
    with tab_summary:
        st.subheader("Counts and Ratios Summary")
        if not summary_csv.exists():
            st.info(f"No counts and ratios summary CSV found at {summary_csv}")
        else:
            df_summary = load_data(summary_csv)
            if not df_summary.empty:
                st.success(f"✅ Loaded {summary_csv.name}")
                st.write(f"Records: {len(df_summary)}")
                st.dataframe(df_summary, use_container_width=True, height=500)
                st.download_button(
                    label="📥 Download Counts & Ratios Summary",
                    data=df_summary.to_csv(index=False).encode("utf-8"),
                    file_name=summary_csv.name,
                    mime="text/csv",
                )

    # ---------------------------------
    # Footer
    # ---------------------------------
    st.markdown("---")
    st.markdown("**Patent Data Analysis Tool** | Data source: PATSTAT")


