# Extract data related to origin country of applt and invt in date range years.
# And store dataframe in database with table name 'patstat_COUNTRY_YEAR1_YEAR2 ALL or 50% '
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from connect_database import create_sqlalchemy_session
from sqlalchemy.orm import aliased
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    String,
    MetaData,
    select,
    or_,
    and_,
    case,
    func,
    distinct,
    and_,
)
from sqlalchemy.sql import func
import matplotlib.pyplot as plt
import ast
import unicodedata
import streamlit as st
import re
from typing import Optional
from typing import Union
import logging
import requests
from scipy.stats import mode  # used to get the most common value/ in inventors counts

# Our functions
from connect_database import create_sqlalchemy_session

from ploting_applicants_inventors_details import (
    plot_appl_invt_ratios,
    plot_appl_invt_counts,
    plot_appl_invt_side_by_side,
    plot_appl_invt_indiv_non_indiv,
    plot_individ_appl_invt_ratios,
    plot_appl_invt_ratios_interactive
)
import config

# Initialize Logger
logger = logging.getLogger(__name__)

# Get constant from config.py
output_dir = Path(config.Config.output_dir)

# Note: Database sessions are created within functions using context managers

# Tables to work with
from models_tables import (
    TLS201_APPLN,
    TLS204_APPLN_PRIOR,
    TLS206_PERSON,
    TLS207_PERS_APPLN,
    TLS226_PERSON_ORIG,
)

# Create aliases for the models
t201 = aliased(TLS201_APPLN)
t204 = aliased(TLS204_APPLN_PRIOR)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)
t226 = aliased(TLS226_PERSON_ORIG)

# Define the query for all applicants/inventors countries within a spesific country_code.
# example country_code = 'NO', but a application can have applicants and inventors from other countries with 'NO'


# Function to fetch family IDs based on country_code and year range
def get_family_ids(country_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    with create_sqlalchemy_session() as db:
        if len(country_code) != 2 or not country_code.isalpha():
            raise ValueError("Country code must be a 2-letter string (e.g., 'NO').")
        if start_year < 1900 or start_year > 2025:
            raise ValueError("Start year must be between 1900 and 2025.")
        if end_year < start_year or end_year > 2025:
            raise ValueError(
                "End year must be greater than or equal to start year and <= 2023."
            )

        query = (
            db.query(t201.appln_id, t201.docdb_family_id, t201.appln_filing_year)
            .join(t207, t201.appln_id == t207.appln_id)
            .join(t206, t207.person_id == t206.person_id)
            .filter(
                t206.person_ctry_code == country_code,
                t201.appln_filing_year.between(start_year, end_year),
            )
            .group_by(t201.appln_id, t201.docdb_family_id, t201.appln_filing_year)
            .order_by(t201.appln_id, t201.appln_filing_year)
        )
        results = query.all()
        # if resutl is empty
        if not results:
            return pd.Series([], dtype="int64")

        df_family_ids = pd.DataFrame(results)
        # Remove duplicates
        df_unique_family_ids = df_family_ids["docdb_family_id"].drop_duplicates()

        # Make df as dataframe instead of serie:
        df_unique_family_ids = pd.DataFrame(
            df_unique_family_ids, columns=["docdb_family_id"]
        )
        return df_unique_family_ids


def get_applicant_inventor(family_ids_list: list[int]):
    """
    Retrieves applicants and inventors for the given family IDs.

    Args:
        family_ids_list (list[int]): List of docdb_family_id values to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing applicant and inventor details.
    """
    try:
        if not family_ids_list or not all(isinstance(i, int) for i in family_ids_list):
            raise ValueError("Family IDs must be a non-empty list of integers.")

        # Using batch for long dataset
        batch_size = config.Config.batch_size
        batches = [
            family_ids_list[i : i + batch_size]
            for i in range(0, len(family_ids_list), batch_size)
        ]

        df_appl_invt = pd.DataFrame()
        all_batches = []
        for batch in batches:
            with create_sqlalchemy_session() as db:
                query = (
                    db.query(
                        t201.docdb_family_id,
                        t201.appln_id,
                        t201.appln_filing_year,
                        t201.appln_auth,
                        t201.appln_nr,
                        t201.docdb_family_size,
                        t201.earliest_publn_date,
                        t201.nb_applicants,
                        t201.nb_inventors,
                        t206.person_ctry_code,
                        t206.person_name,
                        t206.person_id,
                        t206.doc_std_name_id,
                        t206.psn_sector,
                        t207.applt_seq_nr,
                        t207.invt_seq_nr,
                    )
                    .join(t207, t201.appln_id == t207.appln_id)
                    .join(t206, t207.person_id == t206.person_id)
                    .where(t201.docdb_family_id.in_(batch))
                    .order_by(t201.docdb_family_id, t201.appln_id)
                )
                results = query.all()
                df_batch = pd.DataFrame(results).drop_duplicates()
                all_batches.append(df_batch)
                df_appl_invt = (
                    pd.concat(all_batches, ignore_index=True)
                    if all_batches
                    else pd.DataFrame()
                )

    except Exception as e:
        logger.error(f"Error fetching applicant/inventor data: {str(e)}")
        raise

    # Save df_appl_invt to csv for later usage
    df_appl_invt.to_csv(
        Path(config.Config.output_dir) / "df_appl_invt.csv", index=False
    )
    return df_appl_invt


def aggregate_applicants_inventors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to aggregate applicants, inventors, and application IDs for each docdb_family_id.
    Creates new columns with aggregated values, joining multiple entries into comma-separated strings.
    """

    # Function to normalize names (title case + remove accents/special characters)
    def normalize_name(name):
        if pd.isna(name) or not isinstance(name, str):
            return ""
        # Decompose special characters and keep only ASCII
        name = "".join(c for c in unicodedata.normalize("NFKD", name) if c.isascii())
        return name.title()

    # Apply normalization to person_name
    df["person_name_normalized"] = df["person_name"].apply(normalize_name)

    # Extract inventors and applicants
    inventors_df = df[df["invt_seq_nr"] >= 1][
        ["docdb_family_id", "appln_id", "person_name_normalized"]
    ]
    applicants_df = df[df["applt_seq_nr"] >= 1][
        ["docdb_family_id", "appln_id", "person_name_normalized"]
    ]

    # Group and aggregate inventors by docdb_family_id
    inventors_agg = (
        inventors_df.groupby("docdb_family_id")["person_name_normalized"]
        .agg(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"person_name_normalized": "inventors"})
    )

    # Group and aggregate applicants by docdb_family_id
    applicants_agg = (
        applicants_df.groupby("docdb_family_id")["person_name_normalized"]
        .agg(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"person_name_normalized": "applicants"})
    )

    # Aggregate appln_ids by docdb_family_id
    appln_ids_agg = (
        df.groupby("docdb_family_id")["appln_id"]
        .agg(lambda x: ", ".join(map(str, sorted(set(x)))))
        .reset_index()
        .rename(columns={"appln_id": "appln_ids"})
    )

    # Merge the aggregated dataframes
    df_appl_invt_agg = inventors_agg.merge(
        applicants_agg, on="docdb_family_id", how="outer"
    ).merge(appln_ids_agg, on="docdb_family_id", how="outer")

    # Fill NaN values with empty strings

    df_appl_invt_agg["inventors"] = df_appl_invt_agg["inventors"].fillna("")
    df_appl_invt_agg["applicants"] = df_appl_invt_agg["applicants"].fillna("")
    df_appl_invt_agg["appln_ids"] = df_appl_invt_agg["appln_ids"].fillna("")

    return df_appl_invt_agg


# Calculate ratios
def calculate_applicants_inventors_ratios(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
    df_combined_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate applicant, inventor, and combined ratios using pre-calculated counts.

    Args:
        df_applicant_counts (pd.DataFrame): DataFrame with applicant counts
        df_inventor_counts (pd.DataFrame): DataFrame with inventor counts
        df_combined_counts (pd.DataFrame): DataFrame with combined counts

    Returns:
        tuple: (df_applicant_ratios, df_inventor_ratios, df_combined_ratios)
    """

    def calculate_ratio(
        df: pd.DataFrame, count_column: str, ratio_column: str
    ) -> pd.DataFrame:
        # Step 1: Calculate total count per docdb_family_id
        total_counts = (
            df.groupby("docdb_family_id")[count_column]
            .sum()
            .reset_index(name="total_count")
        )

        # Step 2: Merge total counts back into the original DataFrame
        df = pd.merge(df, total_counts, on="docdb_family_id", how="left")

        # Step 3: Calculate the ratio
        df[ratio_column] = df[count_column] / df["total_count"]
        df[ratio_column] = df[ratio_column].fillna(0)

        # Step 4: Keep only relevant columns
        return df[["docdb_family_id", "person_ctry_code", ratio_column]]

    # Step 1: Calculate applicant ratios
    df_applicant_ratios = calculate_ratio(
        df_applicant_counts,
        count_column="applicant_count",
        ratio_column="applicant_ratio",
    )

    # Step 2: Calculate inventor ratios
    df_inventor_ratios = calculate_ratio(
        df_inventor_counts, count_column="inventor_count", ratio_column="inventor_ratio"
    )

    # Step 3: Calculate combined ratios
    df_combined_ratios = calculate_ratio(
        df_combined_counts, count_column="combined_count", ratio_column="combined_ratio"
    )

    return df_applicant_ratios, df_inventor_ratios, df_combined_ratios


###### Calculate Counts
def calculate_applicants_inventors_counts(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate counts of applicants, inventors, and combined per country per docdb_family_id.

    Args:
        df (pd.DataFrame): DataFrame with applicant/inventor data (e.g., from get_applicant_inventor)

    Returns:
        tuple: (df_applicant_counts, df_inventor_counts, df_combined_counts)
        Each DataFrame has columns: docdb_family_id, person_ctry_code, {type}_count
    """

    # Step 1: Clean person_ctry_code
    df["person_ctry_code"] = df["person_ctry_code"].astype(str).str.strip()
    df_cleaned = df[
        df["person_ctry_code"].notna()  # Remove NaN
        & (df["person_ctry_code"] != "")  # Remove empty string
        & (df["person_ctry_code"] != " ")  # Remove single space
        & (df["person_ctry_code"].str.len() > 0)  # Ensure length > 0 after stripping
    ].copy()

    # Step 2: Inventor Counts
    # Step 1: Filter rows with inventors (invt_seq_nr > 0)
    inventor_data = df_cleaned[df_cleaned["invt_seq_nr"] > 0].copy()

    # Step 2: Select the application with max nb_inventors, max nb_applicants, and latest earliest_publn_date
    selected_appln = (
        inventor_data.groupby("docdb_family_id")
        .apply(
            lambda x: x.sort_values(
                by=["nb_inventors", "nb_applicants", "earliest_publn_date"],
                ascending=[False, False, False],
            ).iloc[
                0
            ]  # Take the top row after sorting
        )
        .reset_index(drop=True)
    )

    # Step 3: Extract the selected appln_ids for each docdb_family_id
    selected_appln_ids = selected_appln[["docdb_family_id", "appln_id"]]

    # Step 4: Merge back to get the inventor details for the selected appln_ids
    selected_inventors = inventor_data.merge(
        selected_appln_ids, on=["docdb_family_id", "appln_id"]
    )

    # Step 5: Count distinct inventors (person_id) per country (person_ctry_code) for each family
    df_inventor_counts = (
        selected_inventors.groupby(["docdb_family_id", "person_ctry_code"])["person_id"]
        .nunique()
        .reset_index(name="inventor_count")
    )

    # Display the result (optional: filter for a specific family for verification)
    print("Inventor counts per country for each patent family:")
    print(df_inventor_counts)

    # Step 3: Applicant Counts
    applicant_data = df_cleaned[df_cleaned["applt_seq_nr"] > 0].copy()

    # Step 3: Select the "best" application per family
    # Sort by nb_applicants (descending), nb_inventors (descending), and earliest_publn_date (descending)
    selected_appln = (
        applicant_data.groupby(["docdb_family_id", "person_ctry_code"])
        .apply(
            lambda x: x.sort_values(
                by=["nb_applicants", "nb_inventors", "earliest_publn_date"],
                ascending=[False, False, False],
            ).iloc[
                0
            ]  # Take the top row
        )
        .reset_index(drop=True)
    )

    # Step 4: Extract selected appln_ids for merging
    selected_appln_ids = selected_appln[["docdb_family_id", "appln_id"]]

    # Step 5: Merge back to get all applicant records for the selected applications
    selected_applicants = applicant_data.merge(
        selected_appln_ids, on=["docdb_family_id", "appln_id"]
    )

    # Step 6: Count distinct applicants (person_id) per country (person_ctry_code) for each family
    df_applicant_counts = (
        selected_applicants.groupby(["docdb_family_id", "person_ctry_code"])[
            "person_id"
        ]
        .nunique()
        .reset_index(name="applicant_count")
    )

    # Step 7: Display the results
    print("Number of distinct applicants per patent family:")
    print(df_applicant_counts)

    # Step 4: Combined Counts
    df_combined_counts = (
        pd.concat(
            [
                df_inventor_counts[
                    ["docdb_family_id", "person_ctry_code", "inventor_count"]
                ].rename(columns={"inventor_count": "combined_count"}),
                df_applicant_counts[
                    ["docdb_family_id", "person_ctry_code", "applicant_count"]
                ].rename(columns={"applicant_count": "combined_count"}),
            ]
        )
        .groupby(["docdb_family_id", "person_ctry_code"])
        .sum()
        .reset_index()
    )

    # Calculate ratios
    calculate_applicants_inventors_ratios(
        df_applicant_counts, df_inventor_counts, df_combined_counts
    )

    return df_applicant_counts, df_inventor_counts, df_combined_counts


def classify_entity(name: str, psn_sector: Optional[str] = None) -> str:
    """
    Classify a name as 'INDIVIDUAL' or 'NON_INDIVIDUAL' based on psn_sector or naming patterns.

    Args:
        name (str): The entity name (e.g., from person_name or psn_name).
        psn_sector (Optional[str]): Existing sector value, if available.

    Returns:
        str: 'INDIVIDUAL' or 'NON_INDIVIDUAL'.
    """
    # Define all expected PATSTAT psn_sector categories
    valid_sectors = {
        "INDIVIDUAL": "INDIVIDUAL",
        "COMPANY": "NON_INDIVIDUAL",
        "UNIVERSITY": "NON_INDIVIDUAL",
        "GOV NON-PROFIT": "NON_INDIVIDUAL",
        "GOVERNMENT": "NON_INDIVIDUAL",
        "HOSPITAL": "NON_INDIVIDUAL",
        "UNKNOWN": None,  # Trigger prediction for 'UNKNOWN'
        "": None,  # Trigger prediction for empty string
    }

    # If psn_sector is provided and in valid_sectors (not None), use it
    if (
        psn_sector
        and psn_sector.strip() in valid_sectors
        and valid_sectors[psn_sector.strip()] is not None
    ):
        return valid_sectors[psn_sector.strip()]

    # Predict based on name for missing, empty, 'UNKNOWN', or invalid psn_sector
    name = name.strip().upper()
    non_indiv_keywords = [
        "AS",
        "ASA",
        "INC",
        "LTD",
        "LLC",
        "GMBH",
        "SA",
        "AG",
        "CORP",
        "NV",
        "AB",
        "UNIVERSITY",
        "SCANDINAVIA",
    ]
    # Check for whole-word matches only
    if any(
        re.search(r"\b" + re.escape(keyword) + r"\b", name)
        for keyword in non_indiv_keywords
    ):
        return "NON_INDIVIDUAL"
    if "," in name:
        parts = name.split(",")
        if len(parts) == 2 and all(part.strip() for part in parts):
            return "INDIVIDUAL"
    name = name.strip().upper()
    non_indiv_keywords = [
        "AS",
        "ASA",
        "INC",
        "LTD",
        "LLC",
        "GMBH",
        "SA",
        "AG",
        "CORP",
        "NV",
        "AB",
        "UNIVERSITY",
        "SCANDINAVIA",
    ]
    # Check for whole-word matches only
    if any(
        re.search(r"\b" + re.escape(keyword) + r"\b", name)
        for keyword in non_indiv_keywords
    ):
        return "NON_INDIVIDUAL"
    if "," in name:
        parts = name.split(",")
        if len(parts) == 2 and all(part.strip() for part in parts):
            return "INDIVIDUAL"

    # Split name into parts and check for individual-like patterns
    parts = re.split(r"[,\s]+", name)
    if ("," in name or len(parts) >= 2) and not any(
        part in non_indiv_keywords for part in parts
    ):
        if any(len(part) <= 2 for part in parts) or len(parts) <= 4:
            return "INDIVIDUAL"

    # Default to INDIVIDUAL if no clear non-individual pattern is found
    return "INDIVIDUAL"


def calculate_applicants_inventors_indiv_non_indiv(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate counts of individual inventors, non-individual inventors, non-individual applicants,
    and individual applicants per country per docdb_family_id.

    Args:
        df (pd.DataFrame): DataFrame with applicant/inventor data (e.g., from get_applicant_inventor)

    Returns:
        tuple: (df_invt_indiv_counts, df_invt_non_indiv_counts, df_appl_non_indiv_counts, df_appl_indiv_counts)
        Each DataFrame has columns: docdb_family_id, person_ctry_code, {type}_count
    """

    # Clean and classify
    df["person_ctry_code"] = df["person_ctry_code"].astype(str).str.strip()
    df_cleaned = df[
        df["person_ctry_code"].notna()  # Remove NaN
        & (df["person_ctry_code"] != "")  # Remove empty string
        & (df["person_ctry_code"] != " ")  # Remove single space
        & (df["person_ctry_code"].str.len() > 0)  # Ensure length > 0 after stripping
    ].copy()

    # Select the "best" application per docdb_family_id
    # Filter inventor data
    inventor_data = df_cleaned[df_cleaned["invt_seq_nr"] > 0].copy()
    selected_appln_inventors = (
        inventor_data.groupby("docdb_family_id")
        .apply(
            lambda x: x.sort_values(
                by=["nb_inventors", "nb_applicants", "earliest_publn_date"],
                ascending=[False, False, False],
            ).iloc[
                0
            ]  # Take the top row after sorting
        )
        .reset_index(drop=True)
    )
    selected_appln_ids_inventors = selected_appln_inventors[
        ["docdb_family_id", "appln_id"]
    ]

    # Filter applicant data
    applicant_data = df_cleaned[df_cleaned["applt_seq_nr"] > 0].copy()
    selected_appln_applicants = (
        applicant_data.groupby("docdb_family_id")
        .apply(
            lambda x: x.sort_values(
                by=["nb_applicants", "nb_inventors", "earliest_publn_date"],
                ascending=[False, False, False],
            ).iloc[
                0
            ]  # Take the top row after sorting
        )
        .reset_index(drop=True)
    )
    selected_appln_ids_applicants = selected_appln_applicants[
        ["docdb_family_id", "appln_id"]
    ]

    # Merge back to get the filtered data
    filtered_inventor_data = inventor_data.merge(
        selected_appln_ids_inventors, on=["docdb_family_id", "appln_id"]
    )
    filtered_applicant_data = applicant_data.merge(
        selected_appln_ids_applicants, on=["docdb_family_id", "appln_id"]
    )

    filtered_inventor_data["psn_sector_predicted"] = filtered_inventor_data.apply(
        lambda row: classify_entity(row["person_name"], row["psn_sector"]), axis=1
    )


    filtered_applicant_data["psn_sector_predicted"] = filtered_applicant_data.apply(
        lambda row: classify_entity(row["person_name"], row["psn_sector"]), axis=1
    )

    # Deduplicate entities within the same docdb_family_id and person_ctry_code
    filtered_inventor_data = filtered_inventor_data.drop_duplicates(
        subset=["docdb_family_id", "person_ctry_code", "person_id"]
    )
    filtered_applicant_data = filtered_applicant_data.drop_duplicates(
        subset=["docdb_family_id", "person_ctry_code", "person_id"]
    )

    # Categorize Inventors and Applicants
    # Individual Inventors
    invt_indiv_data = filtered_inventor_data[
        filtered_inventor_data["psn_sector_predicted"] == "INDIVIDUAL"
    ].copy()
    df_invt_indiv_counts = (
        invt_indiv_data.groupby(["docdb_family_id", "person_ctry_code"])["person_id"]
        .nunique()
        .reset_index(name="invt_indiv_count")
    )

    # Non-Individual Inventors
    invt_non_indiv_data = filtered_inventor_data[
        filtered_inventor_data["psn_sector_predicted"] == "NON_INDIVIDUAL"
    ].copy()
    df_invt_non_indiv_counts = (
        invt_non_indiv_data.groupby(["docdb_family_id", "person_ctry_code"])["person_id"]
        .nunique()
        .reset_index(name="invt_non_indiv_count")
    )

    # Non-Individual Applicants
    appl_non_indiv_data = filtered_applicant_data[
        filtered_applicant_data["psn_sector_predicted"] == "NON_INDIVIDUAL"
    ].copy()
    df_appl_non_indiv_counts = (
        appl_non_indiv_data.groupby(["docdb_family_id", "person_ctry_code"])["person_id"]
        .nunique()
        .reset_index(name="appl_non_indiv_count")
    )

    # Individual Applicants
    appl_indiv_data = filtered_applicant_data[
        filtered_applicant_data["psn_sector_predicted"] == "INDIVIDUAL"
    ].copy()
    df_appl_indiv_counts = (
        appl_indiv_data.groupby(["docdb_family_id", "person_ctry_code"])["person_id"]
        .nunique()
        .reset_index(name="appl_indiv_count")
    )

    # Post-processing
    # Ensure no invalid country codes remain
    def filter_invalid_countries(df):
        return df[
            df["person_ctry_code"].notna()
            & (df["person_ctry_code"] != "")
            & (df["person_ctry_code"] != " ")
        ]
    
    # Count of individual/person applicants.
    df_invt_indiv_counts = filter_invalid_countries(df_invt_indiv_counts)
    df_invt_non_indiv_counts = filter_invalid_countries(df_invt_non_indiv_counts)
    df_appl_non_indiv_counts = filter_invalid_countries(df_appl_non_indiv_counts)
    df_appl_indiv_counts = filter_invalid_countries(df_appl_indiv_counts)

    # Return results
    return (
        df_invt_indiv_counts,
        df_invt_non_indiv_counts,
        df_appl_non_indiv_counts,
        df_appl_indiv_counts,
    )


def individ_applicant(
    df_appl_indiv_counts: pd.DataFrame, df_appl_non_indiv_counts: pd.DataFrame
) -> (pd.DataFrame, int, float):
    """
    Calculate the ratio of individual applicants from each country to the total number of applicants per docdb_family_id.
    Additionally, compute dataset-wide statistics:
    - Number of families with at least one individual applicant.
    - Ratio of families with only individual applicants to the total number of families.

    Args:
        df_appl_indiv_counts (pd.DataFrame): DataFrame with columns: docdb_family_id, person_ctry_code, appl_indiv_count
        df_appl_non_indiv_counts (pd.DataFrame): DataFrame with columns: docdb_family_id, person_ctry_code, appl_non_indiv_count

    Returns:
        (pd.DataFrame, int, float): A tuple containing:
            - family_country_indiv_ratio_df: DataFrame with columns: docdb_family_id, person_ctry_code, indiv_applicant_ratio
            - num_families_with_indiv: Number of families with at least one individual applicant
            - ratio_only_indiv: Ratio of families with only individual applicants to the total number of families
    """
    # Merge the DataFrames on docdb_family_id and person_ctry_code with an outer join
    merged_df = pd.merge(
        df_appl_indiv_counts,
        df_appl_non_indiv_counts,
        on=["docdb_family_id", "person_ctry_code"],
        how="outer",
    )

    # Fill NaN values with 0 for counts
    merged_df["appl_indiv_count"] = merged_df["appl_indiv_count"].fillna(0)
    merged_df["appl_non_indiv_count"] = merged_df["appl_non_indiv_count"].fillna(0)

    # Calculate total individual and non-individual applicants per family
    family_totals = (
        merged_df.groupby("docdb_family_id")
        .agg({"appl_indiv_count": "sum", "appl_non_indiv_count": "sum"})
        .reset_index()
    )

    # Number of families with at least one individual applicant
    num_families_with_indiv = (family_totals["appl_indiv_count"] > 0).sum()

    # Number of families with only individual applicants (no non-individual applicants)
    num_families_only_indiv = (
        (family_totals["appl_indiv_count"] > 0)
        & (family_totals["appl_non_indiv_count"] == 0)
    ).sum()

    # Total number of unique families
    total_families = len(family_totals)

    # Ratio of families with only individual applicants to total families
    ratio_only_indiv = (
        num_families_only_indiv / total_families if total_families > 0 else np.nan
    )

    # Calculate total applicants per family
    family_totals["total_applicants"] = (
        family_totals["appl_indiv_count"] + family_totals["appl_non_indiv_count"]
    )

    # Merge total applicants back to the main DataFrame
    merged_df = pd.merge(
        merged_df,
        family_totals[["docdb_family_id", "total_applicants"]],
        on="docdb_family_id",
        how="left",
    )

    # Calculate the per-country ratio: appl_indiv_count / total_applicants
    merged_df["indiv_applicant_ratio"] = np.where(
        merged_df["total_applicants"] > 0,
        merged_df["appl_indiv_count"] / merged_df["total_applicants"],
        np.nan,  # Handle division by zero
    )

    # Select output columns for the DataFrame
    df_indiv_applicant_ratio = merged_df[
        ["docdb_family_id", "person_ctry_code", "indiv_applicant_ratio"]
    ]

    return df_indiv_applicant_ratio, num_families_with_indiv, ratio_only_indiv


def female_invt_ratio(df_appl_invt: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ratio of female inventors for each docdb_family_id and person_ctry_code.

    Args:
        df_appl_invt (pd.DataFrame): DataFrame containing patent applicant and inventor data with columns:
            - docdb_family_id: Unique identifier for patent families
            - person_id: Unique identifier for individuals
            - person_name: Full name of the individual
            - person_ctry_code: Country code of the individual
            - invt_seq_nr: Sequence number indicating inventor status (> 0 for inventors)

    Returns:
        pd.DataFrame: DataFrame with columns:
            - docdb_family_id: Patent family ID
            - person_ctry_code: Country code
            - female_count: Number of female inventors
            - total_count: Total number of inventors with known gender
            - female_ratio: Ratio of female inventors to total inventors
    """
    # Step 1: Filter to include only inventors
    inventors_df = df_appl_invt[df_appl_invt["invt_seq_nr"] > 0].copy()

    # Step 2: Deduplicate inventors per family
    unique_inventors_df = inventors_df.drop_duplicates(
        subset=["docdb_family_id", "person_id"]
    ).copy()

    # Step 3: Define a helper function to infer gender using Genderize.io
    def get_gender(name: str, country: str) -> tuple:
        # Extract first name (handles "Lastname, Firstname" or "Firstname Lastname")
        if "," in name:
            first_name = name.split(",")[-1].strip().split()[0]
        else:
            first_name = name.split()[0]

        # Query Genderize.io API
        url = f"https://api.genderize.io?name={first_name}&country_id={country}"
        response = requests.get(url).json()
        gender = response.get("gender", "unknown")
        probability = response.get("probability", 0)
        if gender is None:
            gender = "unknown"
        return gender, probability

    # Step 4: Apply gender inference with a threshold
    threshold = 0.8
    gender_prob = unique_inventors_df.apply(
        lambda row: get_gender(row["person_name"], row["person_ctry_code"]),
        axis=1,
        result_type="expand",
    )
    unique_inventors_df.loc[:, ["gender", "probability"]] = gender_prob
    unique_inventors_df.loc[:, "classification"] = unique_inventors_df.apply(
        lambda row: row["gender"] if row["probability"] >= threshold else "unknown",
        axis=1,
    )

    # Step 5: Exclude inventors with unknown gender
    filtered_df = unique_inventors_df[
        unique_inventors_df["classification"] != "unknown"
    ]

    # Step 6: Group by family ID and country code
    grouped = filtered_df.groupby(["docdb_family_id", "person_ctry_code"])

    # Count female inventors
    female_counts = (
        grouped["classification"]
        .apply(lambda g: (g == "female").sum())
        .reset_index(name="female_count")
    )

    # Count total inventors with known gender
    total_counts = grouped.size().reset_index(name="total_count")

    # Step 7: Merge counts and calculate the ratio
    ratio_df = pd.merge(
        female_counts, total_counts, on=["docdb_family_id", "person_ctry_code"]
    )
    ratio_df["female_ratio"] = ratio_df["female_count"] / ratio_df["total_count"]
    ratio_df["female_ratio"] = ratio_df["female_ratio"].fillna(0)  # Handle edge cases

    # Select the relevant columns for the output
    df_female_inventor_ratio = ratio_df

    return df_female_inventor_ratio


#######################################
# Parent function: This will start running previews functions over...the call come from main.py
########################################
def get_applicants_inventors_data(country_code: str, start_year: int, end_year: int, range_limit: Optional[int] = None) -> tuple:
    if len(country_code) != 2 or not country_code.isalpha():
        raise ValueError("Country code must be a 2-letter string (e.g., 'NO').")
    if start_year < 1900 or start_year > 2025:
        raise ValueError("Start year must be between 1900 and 2025.")  # Updated to 2025
    if end_year < start_year or end_year > 2025:
        raise ValueError(
            "End year must be >= start year and <= 2025."
        )  # Updated to 2025

    # List of DataFrame names to return
    df_names = [
        "df_unique_family_ids",
        "df_appl_invt",
        "df_appl_invt_agg",
        "df_applicant_ratios",
        "df_inventor_ratios",
        "df_combined_ratios",
        "df_applicant_counts",
        "df_inventor_counts",
        "df_combined_counts",
        "df_invt_indiv_counts",
        "df_invt_non_indiv_counts",
        "df_appl_non_indiv_counts",
        "df_appl_indiv_counts",
        "df_indiv_applicant_ratio",
        "num_families_with_indiv",
        "ratio_only_indiv",
        "df_female_inventor_ratio",
    ]

    df_unique_family_ids = get_family_ids(country_code, start_year, end_year)
    if df_unique_family_ids.empty:
        logger.warning("No family IDs found for the given criteria")
        return tuple(pd.DataFrame() for _ in df_names)

    # For testing purposes
    df_unique_family_ids = df_unique_family_ids[0:range_limit]

    # Convert to list
    family_ids_list = df_unique_family_ids["docdb_family_id"].tolist()

    # Get applicant and inventor data
    df_appl_invt = get_applicant_inventor(family_ids_list)

    # Aggregate names and appln_ids into same rows
    df_appl_invt_agg = aggregate_applicants_inventors(df_appl_invt)

    # Calculate counts
    df_applicant_counts, df_inventor_counts, df_combined_counts = (
        calculate_applicants_inventors_counts(df_appl_invt)
    )

    # Calculate ratios
    df_applicant_ratios, df_inventor_ratios, df_combined_ratios = (
        calculate_applicants_inventors_ratios(
            df_applicant_counts, df_inventor_counts, df_combined_counts
        )
    )

    # Calculate individual/non-individual counts
    (
        df_invt_indiv_counts,
        df_invt_non_indiv_counts,
        df_appl_non_indiv_counts,
        df_appl_indiv_counts,
    ) = calculate_applicants_inventors_indiv_non_indiv(df_appl_invt)

    # Generate plot for individual/non-individual counts
    if all(
        not df.empty
        for df in [
            df_invt_indiv_counts,
            df_invt_non_indiv_counts,
            df_appl_non_indiv_counts,
            df_appl_indiv_counts,
        ]
    ):
        plot_appl_invt_indiv_non_indiv(
            df_invt_indiv_counts,
            df_invt_non_indiv_counts,
            df_appl_non_indiv_counts,
            df_appl_indiv_counts,
            sort_by_country=country_code,
        )
    else:
        logger.warning(
            "One or more individual/non-individual count DataFrames are empty"
        )

    # Calculate individual applicant ratio
    (df_indiv_applicant_ratio, num_families_with_indiv, ratio_only_indiv) = (
        individ_applicant(df_appl_indiv_counts, df_appl_non_indiv_counts)
    )

    # Calculate female inventor ratio
    df_female_inventor_ratio = female_invt_ratio(df_appl_invt)

    # ------------------- Ploting ---------------------

    # Plot ratios
    plot_appl_invt_ratios(
        df_applicant_ratios,
        df_inventor_ratios,
        df_combined_ratios,
        sort_by_country=country_code,
        output_dir=config.Config.output_dir,
    )

    # Plot counts
    plot_appl_invt_counts(
        df_applicant_counts,
        df_inventor_counts,
        df_combined_counts,
        sort_by_country=country_code,
    )

    plot_appl_invt_side_by_side(
        df_applicant_counts, df_inventor_counts, sort_by_country=country_code
    )

    plot_appl_invt_indiv_non_indiv(
        df_invt_indiv_counts,
        df_invt_non_indiv_counts,
        df_appl_non_indiv_counts,
        df_appl_indiv_counts,
        sort_by_country=country_code,
    )

    plot_individ_appl_invt_ratios(
        df_applicant_ratios,
        df_inventor_ratios,
        df_combined_ratios,
        df_indiv_applicant_ratio,
        sort_by_country=country_code,
    )

    plot_appl_invt_ratios_interactive(
        df_applicant_ratios,
        df_inventor_ratios,
        df_combined_ratios,
        sort_by_country=country_code,
    )

    # Return the computed DataFrames
    return (
        df_unique_family_ids,
        df_appl_invt,
        df_appl_invt_agg,
        df_applicant_ratios,
        df_inventor_ratios,
        df_combined_ratios,
        df_applicant_counts,
        df_inventor_counts,
        df_combined_counts,
        df_appl_non_indiv_counts,
        df_appl_indiv_counts,
        df_indiv_applicant_ratio,
        num_families_with_indiv,
        ratio_only_indiv,
        df_female_inventor_ratio,
    )
