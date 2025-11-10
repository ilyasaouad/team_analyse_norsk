"""
Streamlit app for Patent Landscape Analysis
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from io import BytesIO
from main import run_full_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function to create Excel file in memory
def create_excel_file(df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    output.seek(0)
    return output.getvalue()


# Page configuration
st.set_page_config(
    page_title="Patent Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📊 Patent Analysis Tool: Applicants and Inventors")
st.markdown(
    """
This tool analyzes patent landscapes by country and year range.
It fetches data from the patent database and provides comprehensive analysis.
"""
)

st.divider()

# Sidebar for input parameters
st.sidebar.header("⚙️ Analysis Parameters")

country_code = st.sidebar.text_input(
    "Country Code", value="NO", max_chars=2, placeholder="e.g., NO, US, DE"
).upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.sidebar.number_input(
        "Start Year", min_value=1900, max_value=2100, value=2020, step=1
    )

with col2:
    end_year = st.sidebar.number_input(
        "End Year", min_value=1900, max_value=2100, value=2020, step=1
    )

range_limit = st.sidebar.number_input(
    "Max Families to Process (0 = unlimited)",
    min_value=0,
    max_value=10000,
    value=100,
    step=10,
)

st.sidebar.divider()

# Validate inputs
if start_year > end_year:
    st.sidebar.error("❌ Start year must be <= End year")
    st.stop()

if len(country_code) != 2:
    st.sidebar.error("❌ Country code must be 2 letters")
    st.stop()

# Run analysis button
if st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary"):
    st.session_state.run_analysis = True
    st.session_state.country = country_code
    st.session_state.start_year = start_year
    st.session_state.end_year = end_year
    st.session_state.range_limit = range_limit if range_limit > 0 else None

# Main content
if "run_analysis" in st.session_state and st.session_state.run_analysis:

    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        with progress_placeholder.container():
            st.info("⏳ Running analysis... This may take a few minutes.")

        # Run the analysis
        with st.spinner("Fetching and processing data..."):
            df_result = run_full_analysis(
                country_code=st.session_state.country,
                start_year=st.session_state.start_year,
                end_year=st.session_state.end_year,
                range_limit=st.session_state.range_limit,
                save_results=True,
            )

        # Check if results exist
        if df_result.empty:
            st.warning("⚠️ No data found for the specified parameters.")
            st.stop()

        # Get output directory
        base_dir = Path.cwd()
        dir_name = f"DataTables_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}"
        output_dir = base_dir / dir_name

        progress_placeholder.empty()
        status_placeholder.success("✅ Analysis completed successfully!")

        st.divider()

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 Main Table Data",
                "👤 Applicants/Inventors Data",
                "👥 Applicants/Inventors Analysis (Count)",
                "📊 Applicants/Inventors Analysis (Ratios)",
            ]
        )

        # TAB 1: Main Table Data
        with tab1:
            st.subheader("Patent Families - Main Table Data")
            st.markdown(
                """
            This table shows one row per patent family with all enriched information:
            - Key identifiers (family ID, application number, authorities)
            - Technology classification (IPC, CPC, Sector, Field)
            - Patent metadata (filing year, family size, granted status)
            """
            )

            main_table_path = output_dir / "main_table_agg.csv"

            if main_table_path.exists():
                df_main_agg = pd.read_csv(main_table_path)

                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Families", len(df_main_agg))
                with col2:
                    st.metric(
                        "Granted Patents",
                        (
                            (df_main_agg["granted"] == "Y").sum()
                            if "granted" in df_main_agg.columns
                            else 0
                        ),
                    )
                with col3:
                    st.metric(
                        "Unique Sectors",
                        (
                            df_main_agg["sector"].nunique()
                            if "sector" in df_main_agg.columns
                            else 0
                        ),
                    )
                with col4:
                    st.metric(
                        "Avg Family Size",
                        (
                            df_main_agg["docdb_family_size"].mean()
                            if "docdb_family_size" in df_main_agg.columns
                            else 0
                        ),
                    )

                st.divider()

                # Display table with pagination
                st.dataframe(df_main_agg, use_container_width=True, height=500)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_main_agg.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"main_table_agg_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.csv",
                        mime="text/csv",
                    )
                with col2:
                    excel_data = create_excel_file(df_main_agg, "Main Table Data")
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=f"main_table_agg_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.error(f"File not found: {main_table_path}")

        # TAB 2: Applicants/Inventors Data (Extracted)
        with tab2:
            st.subheader("Applicants & Inventors - Extracted Data by Family")
            st.markdown(
                """
            This table shows detailed information about applicants and inventors aggregated by patent family:
            - Number of applicants and inventors per family
            - All countries represented (comma-separated)
            - All person names involved (comma-separated)
            - Sectors represented in the family
            - Separated by role (Applicants / Inventors)
            """
            )

            extracted_agg_path = output_dir / "applicants_inventors_extracted_agg.csv"

            if extracted_agg_path.exists():
                df_extracted_agg = pd.read_csv(extracted_agg_path)

                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df_extracted_agg))
                with col2:
                    st.metric(
                        "Total Applicants",
                        (
                            df_extracted_agg["nb_applicants"].sum()
                            if "nb_applicants" in df_extracted_agg.columns
                            else 0
                        ),
                    )
                with col3:
                    st.metric(
                        "Total Inventors",
                        (
                            df_extracted_agg["nb_inventors"].sum()
                            if "nb_inventors" in df_extracted_agg.columns
                            else 0
                        ),
                    )
                with col4:
                    st.metric(
                        "Unique Families",
                        (
                            df_extracted_agg["docdb_family_id"].nunique()
                            if "docdb_family_id" in df_extracted_agg.columns
                            else 0
                        ),
                    )

                st.divider()

                # Display table
                st.dataframe(df_extracted_agg, use_container_width=True, height=500)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_extracted_agg.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"applicants_inventors_extracted_agg_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.csv",
                        mime="text/csv",
                    )
                with col2:
                    excel_data = create_excel_file(
                        df_extracted_agg, "Applicants/Inventors Data"
                    )
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=f"applicants_inventors_extracted_agg_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.info("📌 Applicants/Inventors data file not found.")

        # TAB 3: Applicants/Inventors Analysis (Count)
        with tab3:
            st.subheader("Applicants & Inventors - Analysis by Count")
            st.markdown(
                """
            This table shows the raw counts of applicants and inventors per country per family.
            Use this data for custom analysis in Excel.
            """
            )

            # Try to find the file
            possible_paths = [
                output_dir / "applicants_inventors_by_country.csv",
                output_dir / "applicants_inventors_counts.csv",
            ]

            df_counts = None
            counts_path = None

            for path in possible_paths:
                if path.exists():
                    df_counts = pd.read_csv(path)
                    counts_path = path
                    break

            if df_counts is not None:
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Unique Countries", df_counts["person_ctry_code"].nunique()
                    )
                with col2:
                    st.metric(
                        "Total Applicants",
                        (
                            df_counts["applicant_count"].sum()
                            if "applicant_count" in df_counts.columns
                            else 0
                        ),
                    )
                with col3:
                    st.metric(
                        "Total Inventors",
                        (
                            df_counts["inventor_count"].sum()
                            if "inventor_count" in df_counts.columns
                            else 0
                        ),
                    )

                st.divider()

                # Display table
                st.dataframe(df_counts, use_container_width=True, height=500)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_counts.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"applicants_inventors_counts_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.csv",
                        mime="text/csv",
                    )
                with col2:
                    excel_data = create_excel_file(df_counts, "Analysis Count")
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=f"applicants_inventors_counts_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.info("📌 Applicants/Inventors count file not found.")

        # TAB 4: Applicants/Inventors Analysis (Ratios)
        with tab4:
            st.subheader("Applicants & Inventors - Analysis by Ratios")
            st.markdown(
                """
            This table shows ratios and detailed analysis of applicants and inventors by country per family.
            """
            )

            # Try to find the analysis file
            analysis_path = output_dir / "applicants_inventors_analysis.csv"

            if analysis_path.exists():
                df_analysis = pd.read_csv(analysis_path)

                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(df_analysis))
                with col2:
                    st.metric(
                        "Unique Countries",
                        (
                            df_analysis["person_ctry_code"].nunique()
                            if "person_ctry_code" in df_analysis.columns
                            else 0
                        ),
                    )
                with col3:
                    st.metric(
                        "Avg Applicant Ratio",
                        (
                            df_analysis["applicant_ratio"].mean()
                            if "applicant_ratio" in df_analysis.columns
                            else 0
                        ),
                    )

                st.divider()

                # Display table
                st.dataframe(df_analysis, use_container_width=True, height=500)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_analysis.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"applicants_inventors_analysis_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.csv",
                        mime="text/csv",
                    )
                with col2:
                    excel_data = create_excel_file(df_analysis, "Analysis Ratios")
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=f"applicants_inventors_analysis_{st.session_state.country}_{st.session_state.start_year}_{st.session_state.end_year}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.info("📌 Applicants/Inventors analysis file not found.")

        st.divider()

        # Reset button
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.run_analysis = False
            st.rerun()

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ Error during analysis: {str(e)}")
        logger.exception(e)

else:
    # Initial page when no analysis has been run
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "🔧 **How to Use:**\n\n1. Enter parameters in the sidebar\n2. Click 'Run Analysis'\n3. View results in the tabs"
        )

    with col2:
        st.info(
            "📂 **Output Files:**\n\n- Main table with classifications\n- Applicant/inventor data by country\n- Ready for Excel analysis"
        )

    st.markdown("---")
    st.markdown(
        """
    ### About This Tool
    
    This Patent Landscape Analysis tool helps you:
    - **Fetch** patent family data from the database
    - **Enrich** with technology sectors and fields (IPC mapping)
    - **Analyze** applicants and inventors by country
    - **Export** data for further analysis in Excel
    
    **Output Location:** `DataTables_[COUNTRY]_[YEAR]_[YEAR]/`
    """
    )
