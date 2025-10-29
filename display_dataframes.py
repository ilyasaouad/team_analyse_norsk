import streamlit as st
import pandas as pd
from pathlib import Path

from config import Config

# Simple password gate
password = st.text_input("Enter password:", type="password")
if password != st.secrets["app_password"]:
    if password:
        st.error("Wrong password")
    st.stop()

# Optional visual confirmation
st.success("Access granted")

# Page configuration
st.set_page_config(
    page_title="Patent Data Viewer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Patent Applicant/Inventor Data Viewer")

@st.cache_data
def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

# Sidebar configuration
st.sidebar.header("Data Source")
default_dir = Path(Config.output_dir)
available_runs = sorted(
    [p for p in Path.cwd().glob("DataTables_*") if p.is_dir()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
run_options = [str(p) for p in available_runs]
placeholder = str(default_dir if default_dir.exists() else Path.cwd())

selected_option = st.sidebar.selectbox(
    "Available run directories",
    run_options,
    index=0 if run_options else 0,
) if run_options else placeholder

custom_dir = st.sidebar.text_input(
    "Output directory (override if needed)",
    value=selected_option if run_options else placeholder,
)

base_path = Path(custom_dir).expanduser()
families_csv = base_path / "data" / "applicants_inventors" / "unique_family_ids.csv"
details_csv = base_path / "data" / "applicants_inventors" / "applicant_inventor_details.csv"

if not base_path.exists():
    st.error(f"❌ Directory not found: {base_path}")
    st.stop()

tab_family, tab_details = st.tabs(["🧬 Family IDs", "👤 Applicant/Inventor Details"])

with tab_family:
    st.subheader("Unique DOCDB Family IDs")
    if not families_csv.exists():
        st.info(f"No family ID CSV found at {families_csv}")
    else:
        df_family = load_data(families_csv)
        st.success(f"✅ Loaded {families_csv.name}")
        st.write(f"Records: {len(df_family)}")
        st.dataframe(df_family, use_container_width=True, height=500)
        st.download_button(
            label="📥 Download Family IDs",
            data=df_family.to_csv(index=False).encode("utf-8"),
            file_name=families_csv.name,
            mime="text/csv",
        )

with tab_details:
    st.subheader("Applicant / Inventor Details")
    if not details_csv.exists():
        st.info(f"No applicant/inventor CSV found at {details_csv}")
    else:
        df_details = load_data(details_csv)
        st.success(f"✅ Loaded {details_csv.name}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df_details))
        with col2:
            st.metric("Total Columns", len(df_details.columns))
        with col3:
            unique_families = df_details.get("docdb_family_id", pd.Series(dtype="int64")).nunique()
            st.metric("Unique Family IDs", int(unique_families))
        with col4:
            unique_persons = df_details.get("person_name", pd.Series(dtype="object")).nunique()
            st.metric("Unique Persons", int(unique_persons))

        data_tab, filter_tab, stats_tab, download_tab = st.tabs(
            ["📋 Data Table", "🔍 Filters", "📈 Statistics", "💾 Download"]
        )

        with data_tab:
            st.dataframe(df_details, use_container_width=True, height=600)

        with filter_tab:
            filtered_df = df_details.copy()

            if "person_ctry_code" in filtered_df.columns:
                countries = ["All"] + sorted(filtered_df["person_ctry_code"].dropna().unique().tolist())
                selected_country = st.selectbox("Filter by Country", countries)
                if selected_country != "All":
                    filtered_df = filtered_df[filtered_df["person_ctry_code"] == selected_country]

            if "role" in filtered_df.columns:
                roles = ["All"] + sorted(filtered_df["role"].dropna().unique().tolist())
                selected_role = st.selectbox("Filter by Role", roles)
                if selected_role != "All":
                    filtered_df = filtered_df[filtered_df["role"] == selected_role]

            if "person_name" in filtered_df.columns:
                search_term = st.text_input("Search by Person Name", "")
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df["person_name"].str.contains(search_term, case=False, na=False)
                    ]

            st.write(f"Showing {len(filtered_df)} of {len(df_details)} records")
            st.dataframe(filtered_df, use_container_width=True, height=500)

        with stats_tab:
            col_left, col_right = st.columns(2)

            with col_left:
                if "person_ctry_code" in df_details.columns:
                    st.write("**Top Countries by Records**")
                    country_counts = df_details["person_ctry_code"].value_counts().head(10)
                    st.bar_chart(country_counts)

            with col_right:
                if "role" in df_details.columns:
                    st.write("**Distribution by Role**")
                    role_counts = df_details["role"].value_counts()
                    st.bar_chart(role_counts)

            st.write("**Column Information**")
            col_info = pd.DataFrame(
                {
                    "Column": df_details.columns,
                    "Non-Null Count": [df_details[col].count() for col in df_details.columns],
                    "Null Count": [df_details[col].isna().sum() for col in df_details.columns],
                    "Data Type": [df_details[col].dtype for col in df_details.columns],
                }
            )
            st.dataframe(col_info, use_container_width=True)

        with download_tab:
            st.download_button(
                label="📥 Download Full Dataset",
                data=df_details.to_csv(index=False).encode("utf-8"),
                file_name=details_csv.name,
                mime="text/csv",
            )

            if "filtered_df" in locals() and len(filtered_df) < len(df_details):
                st.download_button(
                    label="📥 Download Filtered Dataset",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_applicant_inventor_details.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.markdown("**Patent Data Analysis Tool** | Data source: PATSTAT")
