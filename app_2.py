import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from matplotlib import pyplot as plt
import streamlit as st
import plotly

# Our functions
from get_applicants_inventors_details_old import get_applicants_inventors_data
from connect_database import create_sqlalchemy_session
from config import Config  
from prompts import PROMPTS
from llm_analyse import analyze_dataframe
from ploting_applicants_inventors_details import plot_appl_invt_ratios_interactive

# Setup logging
def setup_logging():
    """Configure logging for the application"""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

# Function to create output directory
def create_data_folder(country_code, start_year, end_year, working_dir):
    """
    Create a folder for storing data and return the output directory path.
    """
    folder_name = f"dataTable_{country_code}_{start_year}_{end_year}"
    output_dir = working_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def main():
    st.title("Patent Data Analysis")

    # Input fields
    country_code = st.text_input("Country Code", value="NO")
    start_year = st.number_input("Start Year", min_value=1900, max_value=2100, value=2020)
    end_year = st.number_input("End Year", min_value=1900, max_value=2100, value=2020)

    # Define working directory
    working_dir = Path("C:/Users/iao/Desktop/Patstat_TIP/Patent_family/applicants_inventors_analyse/")

    # Button to process data
    if st.button("Process Data"):
        with st.spinner("Processing patent data..."):
            try:
                # Log start of processing
                logger.info(
                    f"Processing data for country: {country_code}, years: {start_year}-{end_year}"
                )

                # Create output directory
                output_dir = create_data_folder(
                    country_code, start_year, end_year, working_dir
                )

                # Update Config with new settings
                Config.update(
                    output_dir=str(output_dir),
                    country_code=country_code,
                    start_year=start_year,
                    end_year=end_year,
                )

                # Default value for range_limit Optional: NONE
                range_limit = 15  # Set a range limit for number of family_id to use for testing
            
                dfs = get_applicants_inventors_data(
                    Config.country_code, Config.start_year, Config.end_year, range_limit
                )

                # Unpack the tuple
                (
                    df_unique_family_ids,
                    df_appl_invt,
                    df_appl_invt_agg,
                    df_applicant_ratios,
                    df_inventor_ratios,
                    df_combined_ratios,
                    df_applicant_counts,
                    df_inventor_counts,
                    df_combined_counts,
                    df_appl_non_indiv_counts, # organitation applicant counts
                    df_appl_indiv_counts,     # individual applicant counts
                    df_indiv_applicant_ratio,  # individual applicant ratio
                    num_families_with_indiv,   # number of families with at least one individual applicants
                    ratio_only_indiv,
                    df_female_inventor_ratio,
                ) = dfs

                # Define DataFrame names
                df_names = [
                    "unique_family_ids",
                    "appl_invt",
                    "appl_invt_agg",
                    "applicant_ratios",
                    "inventor_ratios",
                    "combined_ratios",
                    "applicant_counts",
                    "inventor_counts",
                    "combined_counts",
                    "appl_non_indiv_counts",
                    "appl_indiv_counts",
                    "indiv_applicant_ratio",
                    "num_families_with_indiv",
                    "ratio_only_indiv",
                    "female_inventor_ratio",
                ]

                # Save DataFrames to CSV
                csv_output_dir = output_dir / "data" / "applicants_inventors"
                csv_output_dir.mkdir(parents=True, exist_ok=True)
                for i, (df_item, name) in enumerate(zip(dfs, df_names)):
                    filepath = csv_output_dir / f"{name}.csv"
                    if isinstance(df_item, pd.DataFrame):
                        df_item.to_csv(filepath, index=False)
                        logger.info(f"Saved DataFrame '{name}' to {filepath}")
                    else:
                        value_df = pd.DataFrame({"value": [df_item]})
                        value_df.to_csv(filepath, index=False)
                        logger.info(f"Saved value '{name}' to {filepath}")

                # Analyse inventor applicant counts
                dataframes_list = [df_applicant_counts, df_inventor_counts, df_combined_counts]
                dataframe_names = ["applicant_counts", "inventor_counts", "combined_counts"]

                prompt_name = "applicants_inventors_count"
                # Get analysis for all dataframes together
                analysis_result = analyze_dataframe(
                dataframes_list,
                dataframe_names,
                prompt_name,
                Config.country_code
                )

                # Save individual analyses
                txt_output_dir = output_dir / "analyse" / "applicants_inventors"
                txt_output_dir.mkdir(parents=True, exist_ok=True)

                for individual in analysis_result["individual_responses"]:
                    df_name = individual["df_name"]
                    response = individual["response"]
                    filepath = txt_output_dir / f"{df_name}_analysis.txt"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(response)
                    logger.info(f"Saved analysis for '{df_name}' to {filepath}")

                # Save summary counts
                summary_filepath = txt_output_dir / "summary_applicants_inventors_counts.txt"
                with open(summary_filepath, "w", encoding="utf-8") as f:
                    f.write(analysis_result["summary"])
                logger.info(f"Saved summary counts to {summary_filepath}")
   
                # Analyse inventor applicant ratios
                dataframes_list = [df_applicant_ratios, df_inventor_ratios, df_combined_ratios]
                dataframe_names = ["applicant_ratios", "inventor_ratios", "combined_ratios"]

                prompt_name = "applicants_inventors_ratio"
                # Get analysis for all dataframes together
                analysis_result = analyze_dataframe(
                dataframes_list,
                dataframe_names,
                prompt_name,
                Config.country_code
                )

                # Save individual analyses
                txt_output_dir = output_dir / "analyse" / "applicants_inventors"
                txt_output_dir.mkdir(parents=True, exist_ok=True)

                for individual in analysis_result["individual_responses"]:
                    df_name = individual["df_name"]
                    response = individual["response"]
                    filepath = txt_output_dir / f"{df_name}_analysis.txt"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(response)
                    logger.info(f"Saved analysis for '{df_name}' to {filepath}")

                # Save summary ratios
                summary_filepath = txt_output_dir / "summary_applicants_inventors_ratios.txt"
                with open(summary_filepath, "w", encoding="utf-8") as f:
                    f.write(analysis_result["summary"])
                logger.info(f"Saved summary ratios to {summary_filepath}")

                ################
                # Analyse individual applicant counts and ratios
                #################
                dataframes_list = [df_appl_indiv_counts, df_indiv_applicant_ratio]
                dataframe_names = ["appl_indiv_counts", "indiv_applicant_ratios",]

                prompt_name = "inventors_as_applicants_ratio"
                # Get analysis for all dataframes together
                analysis_result = analyze_dataframe(
                dataframes_list,
                dataframe_names,
                prompt_name,
                Config.country_code
                )

                # Save individual analyses
                txt_output_dir = output_dir / "analyse" / "applicants_inventors"
                txt_output_dir.mkdir(parents=True, exist_ok=True)

                for individual in analysis_result["individual_responses"]:
                    df_name = individual["df_name"]
                    response = individual["response"]
                    filepath = txt_output_dir / f"{df_name}_analysis.txt"
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(response)
                    logger.info(f"Saved analysis for '{df_name}' to {filepath}")

                # Save summary ratios
                summary_filepath = txt_output_dir / "summary_inventors_as_applicants_ratio.txt"
                with open(summary_filepath, "w", encoding="utf-8") as f:
                    f.write(analysis_result["summary"])
                logger.info(f"Saved summary ratios to {summary_filepath}")



                # Define the directory where plots are saved
                plots_dir = Path(Config.output_dir) / "plots" / "applicants_inventors"

                # Updated DataFrames dictionary with singular names to match file prefixes
                dataframes = {
                    "applicant_ratios": ("Applicant Ratios", df_applicant_ratios),
                    "inventor_ratios": ("Inventor Ratios", df_inventor_ratios),
                    "combined_ratios": ("Combined Ratios", df_combined_ratios),
                    "applicant_counts": ("Applicant Counts", df_applicant_counts),
                    "inventor_counts": ("Inventor Counts", df_inventor_counts),
                    "combined_counts": ("Combined Counts", df_combined_counts),
                    "appl_invt_indiv_non_indiv_pos_neg_counts": ("Applicant and Inventor, Individual non-individual Counts",df_appl_indiv_counts),
                    "appl_indiv_counts": ("Applicant Individual Counts", df_appl_indiv_counts),
                    "indiv_applicant_ratios": ("Individual Applicant Ratio", df_indiv_applicant_ratio),
                    "female_inventor_ratios": ("Female Inventor Ratio",df_female_inventor_ratio),
                }

                # Dictionary mapping DataFrames to their plots (filenames without paths)
                plot_mappings = {
                    "applicant_counts": [
                        "applicant_counts.png",
                    ],
                    "inventor_counts": [
                        "inventor_counts.png",
                    ],
                    "combined_counts": [
                        "combined_counts.png",
                        "inventor_counts_side_by_side_applicant_counts.png",
                    ],
                    "applicant_ratios": [
                        "applicant_ratios.png",
                    ],
                    "inventor_ratios": [
                        "inventor_ratios.png",
                    ],
                    "combined_ratios": [
                        "combined_ratios.png",
                    ],
                    "appl_invt_indiv_non_indiv_pos_neg_counts": [
                        "inventor_applicant_indiv_non_indiv_pos_neg_plot.png",
                    ],
                    "appl_non_indiv_counts": [
                        "inventor_applicant_indiv_non_indiv.png",
                    ],
                    "appl_applicant_ratios": [
                        "indiv_applicant_ratio.png",
                    ],
                    "female_inventor_ratios": [
                        "female_inventor_ratio.png",
                    ],
                }

                # Dictionary mapping DataFrames to their analyses (filenames without paths)
                analysis_mappings = {
                    "applicant_counts": [
                        "applicant_counts_analysis.txt",   
                    ],
                    "inventor_counts": [
                        "inventor_counts_analysis.txt",   
                    ],
                    "combined_counts": [
                        "combined_counts_analysis.txt",
                        "summary_applicants_inventors_counts.txt"   
                    ],
                    "applicant_ratios": [
                        "applicant_ratios_analysis.txt",   
                    ],
                    "inventor_ratios": [
                        "inventor_ratios_analysis.txt",   
                    ],
                    "combined_ratios": [
                        "combined_ratios_analysis.txt",
                        "summary_applicants_inventors_ratios.txt"   
                    ],

                    "appl_indiv_counts": [
                        "appl_indiv_counts.txt",  
                    ],
                    "indiv_applicant_ratios": [
                        "applicant_indiv_ratio.txt", 
                    ],
                }

                # Helper function to load file content
                def load_file_content(file_path: Path, encoding="utf-8") -> str:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            return f.read()
                    except Exception as e:
                        st.error(f"Error loading {file_path}: {e}")
                        return ""

                # Function to display DataFrame section
                def display_dataframe_section(display_name: str, df):
                    st.write(f"#### {display_name} Data")
                    if df is not None and not df.empty:
                        st.dataframe(df)
                    else:
                        st.warning(f"No data available for '{display_name}'.")

                # Function to display Plots section
                def display_plots_section(display_name: str, plot_files: list):
                    st.write(f"#### {display_name} Plots")
                    if plot_files:
                        for plot_filename in plot_files:
                            plot_file = plots_dir / plot_filename
                            if plot_file.exists():
                                if plot_file.suffix == ".png":
                                    st.image(
                                        str(plot_file),
                                        caption=plot_file.stem.replace("_", " ").title(),
                                        use_container_width=True,
                                    )
                                elif plot_file.suffix == ".html":
                                    html_content = load_file_content(plot_file)
                                    if html_content:
                                        st.components.v1.html(html_content, height=600, scrolling=True)
                                        st.caption(plot_file.stem.replace("_", " ").title())
                            else:
                                st.info(f"Plot '{plot_filename}' not found in {plots_dir}.")
                    else:
                        st.info(f"No plots assigned to '{display_name}'.")

                # Function to display Analyses section
                def display_analyses_section(display_name: str, analysis_files: list):
                    st.write(f"#### {display_name} Analyses")
                    if analysis_files:
                        for analysis_filename in analysis_files:
                            analysis_file = txt_output_dir / analysis_filename
                            if analysis_file.exists():
                                analysis_content = load_file_content(analysis_file)
                                if analysis_content:
                                    st.text_area(
                                        f"Analysis: {analysis_file.stem.replace('_', ' ').title()}",
                                        value=analysis_content,
                                        height=200,
                                        key=f"analysis_{analysis_filename}"
                                    )
                            else:
                                st.info(f"Analysis '{analysis_filename}' not found in {txt_output_dir}.")
                    else:
                        st.info(f"No analyses assigned to '{display_name}'.")

                # Parent function to display all sections
                def display_data_section(df_name: str, display_name: str, df, plot_files: list, analysis_files: list):
                    with st.expander(f"### {display_name}", expanded=False):
                        display_dataframe_section(display_name, df)
                        display_plots_section(display_name, plot_files)
                        display_analyses_section(display_name, analysis_files)

                # Main function to display all data
                def display_all_data():
                    st.title("Data, Plots, and Analyses Explorer")
                    for df_name, (display_name, df) in dataframes.items():
                        plot_files = plot_mappings.get(df_name, [])
                        analysis_files = analysis_mappings.get(df_name, [])
                        display_data_section(df_name, display_name, df, plot_files, analysis_files)

                # Display all data
                display_all_data()

            except Exception as e:
                logger.error(f"An error occurred: {e}", exc_info=True)
                st.error(f"An error occurred: {e}")
# Run the app
if __name__ == "__main__":
    main()
