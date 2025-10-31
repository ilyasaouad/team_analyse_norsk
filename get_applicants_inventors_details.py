import logging
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy.orm import aliased

from config import Config
from connect_database import create_sqlalchemy_session
from models_tables import (
    TLS201_APPLN,
    TLS206_PERSON,
    TLS207_PERS_APPLN,
)

# Using aliased is good practice, especially for complex queries or self-joins.
t201 = aliased(TLS201_APPLN)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)

logger = logging.getLogger(__name__)


def get_family_ids(country_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetches unique DOCDB family IDs for patent applications where at least one
    associated person (applicant or inventor) matches the provided country code
    within the specified filing year range.

    Args:
        country_code: A 2-letter ISO country code (e.g., 'US', 'DE').
        start_year: The starting year of the filing date range (inclusive).
        end_year: The ending year of the filing date range (inclusive).

    Returns:
        A DataFrame containing a single column 'docdb_family_id' with unique IDs.
        Returns an empty DataFrame if no matches are found.
    """
    with create_sqlalchemy_session() as db:
        # --- Input Validation ---
        if len(country_code) != 2 or not country_code.isalpha():
            raise ValueError("Country code must be a 2-letter string (e.g., 'US').")
        if not (1900 <= start_year <= 2025):
            raise ValueError("Start year must be between 1900 and 2025.")
        if not (start_year <= end_year <= 2025):
            # Refactor: Corrected error message to match validation logic.
            raise ValueError(
                "End year must be >= start year and <= 2025."
            )

        # Refactor: Optimized query to get distinct family IDs directly from the DB.
        # This is much more efficient than fetching all rows and deduplicating in pandas.
        query = (
            db.query(t201.docdb_family_id)
            .join(t207, t201.appln_id == t207.appln_id)
            .join(t206, t207.person_id == t206.person_id)
            .filter(
                t206.person_ctry_code == country_code,
                t201.appln_filing_year.between(start_year, end_year),
            )
            .distinct()  # Let the database handle the deduplication.
            .order_by(t201.docdb_family_id)
        )

        results = query.all()

        if not results:
            # Refactor: Use a more standard way to create an empty DataFrame with the correct column.
            return pd.DataFrame(columns=["docdb_family_id"])

        # Refactor: The results are already distinct, so no need for drop_duplicates().
        df_family_ids = pd.DataFrame(results, columns=["docdb_family_id"])
        return df_family_ids


def get_applicant_inventor(family_ids_list: list[int]) -> pd.DataFrame:
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

        # Using batch for long dataset divid family_id into batch here 200 from config 
        batch_size = Config.batch_size
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
                    .filter(t201.docdb_family_id.in_(batch))
                    .order_by(t201.docdb_family_id, t201.appln_id)
                )

                results = query.all()
                if results:
                    df_batch = pd.DataFrame(results).drop_duplicates()
                    all_batches.append(df_batch)

        # Concatenation 'outside' the loop for efficiency
        df_appl_invt = (
            pd.concat(all_batches, ignore_index=True)
            if all_batches
            else pd.DataFrame()
        )


    except Exception as e:
        logger.error(f"Error fetching applicant/inventor data: {str(e)}")
        raise

    output_dir = Path(Config.output_dir) / "data" / "applicants_inventors"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "applicant_inventor_details.csv"
    df_appl_invt.to_csv(output_path, index=False)
    return df_appl_invt


#####################################################
# --------------- Parent function ---------------------
# Call other functions
#####################################################


def get_applicants_inventors_data(
    country_code: str,
    start_year: int,
    end_year: int,
    range_limit: Optional[int] = None,  # For testing with smaller samples like 10.
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves patent family IDs and their associated applicant/inventor details
    for a given country and filing year range.

    Args:
        country_code: A 2-letter ISO country code.
        start_year: The start of the filing year range.
        end_year: The end of the filing year range.
        range_limit: An optional limit on the number of family IDs to process.

    Returns:
        A tuple containing two DataFrames:
        1. A DataFrame of unique 'docdb_family_id's.
        2. A DataFrame with details for each applicant/inventor associated
           with those families, including name, country, and role.
    """
    # --- Input Validation (re-using logic from get_family_ids) ---
    if len(country_code) != 2 or not country_code.isalpha():
        raise ValueError("Country code must be a 2-letter string (e.g., 'US').")
    if not (1900 <= start_year <= 2025):
        raise ValueError("Start year must be between 1900 and 2025.")
    if not (start_year <= end_year <= 2025):
        raise ValueError("End year must be >= start year and <= 2025.")

    # Step 1: Get the unique family IDs.
    df_family_ids = get_family_ids(country_code, start_year, end_year)

    if df_family_ids.empty:
        logger.warning("No family IDs found for the given criteria.")
        return df_family_ids, pd.DataFrame() # Return empty DFs

    # Refactor: Correctly handle the optional range_limit.
    if range_limit is not None and range_limit > 0:
        df_family_ids = df_family_ids.head(range_limit)
        logger.info(f"Applying range limit of {range_limit} family IDs.")

    family_ids_list = df_family_ids["docdb_family_id"].tolist()

    # Step 2: Fetch applicant/inventor details for the found family IDs.
    with create_sqlalchemy_session() as db:
        details_query = (
            db.query(
                t201.docdb_family_id,
                t206.person_name,
                t206.person_ctry_code,
                t207.applt_seq_nr,
                t207.invt_seq_nr,
            )
            .join(t207, t201.appln_id == t207.appln_id)
            .join(t206, t207.person_id == t206.person_id)
            .filter(t201.docdb_family_id.in_(family_ids_list))
            .order_by(t201.docdb_family_id, t206.person_name)
        )

        details_results = details_query.all()

        if not details_results:
            logger.info(
                "No applicant/inventor details found for the selected family IDs."
            )
            return df_family_ids, pd.DataFrame()

        df_details = pd.DataFrame(
            details_results,
            columns=[
                "docdb_family_id",
                "person_name",
                "person_ctry_code",
                "applt_seq_nr",
                "invt_seq_nr",
            ],
        )

        # Refactor: Add a 'role' column to distinguish between Applicant and Inventor.
        # A person is an inventor if invt_seq_nr is not NULL, and an applicant
        # if applt_seq_nr is not NULL. They can be both.
        df_details["role"] = ""
        df_details.loc[df_details["invt_seq_nr"].notna(), "role"] += "Inventor"
        df_details.loc[df_details["applt_seq_nr"].notna(), "role"] += "Applicant"
        df_details["role"] = df_details["role"].str.replace("InventorApplicant", "Inventor, Applicant")

        logger.info(
            f"Retrieved details for {len(df_details)} persons across {len(df_family_ids)} families."
        )

        return df_family_ids, df_details
#####################################################
# --------------- Analysis Functions ----------------
#####################################################


def calculate_applicants_inventors_counts(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate counts of applicants, inventors, and combined per country per docdb_family_id.
    """
    if df.empty:
        return (
            pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", "applicant_count"]
            ),
            pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", "inventor_count"]
            ),
            pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", "combined_count"]
            ),
        )

    # Clean country codes
    df["person_ctry_code"] = df["person_ctry_code"].astype(str).str.strip()
    df = df[df["person_ctry_code"].notna() & (df["person_ctry_code"] != "")].copy()

    # Helper to select best application per family
    def select_best_application(
        data: pd.DataFrame, sort_cols: list[str]
    ) -> pd.DataFrame:
        return (
            data.groupby("docdb_family_id")
            .apply(
                lambda x: x.sort_values(
                    by=sort_cols, ascending=[False, False, False]
                ).iloc[0]
            )
            .reset_index(drop=True)
        )

    # Inventor counts
    inventor_data = df[df["invt_seq_nr"] > 0].copy()
    if not inventor_data.empty:
        selected_inv = select_best_application(
            inventor_data, ["nb_inventors", "nb_applicants", "earliest_publn_date"]
        )
        selected_inv_ids = selected_inv[["docdb_family_id", "appln_id"]]
        selected_inventors = inventor_data.merge(
            selected_inv_ids, on=["docdb_family_id", "appln_id"]
        )
        df_inventor_counts = (
            selected_inventors.groupby(["docdb_family_id", "person_ctry_code"])[
                "person_id"
            ]
            .nunique()
            .reset_index(name="inventor_count")
        )
    else:
        df_inventor_counts = pd.DataFrame(
            columns=["docdb_family_id", "person_ctry_code", "inventor_count"]
        )

    # Applicant counts
    applicant_data = df[df["applt_seq_nr"] > 0].copy()
    if not applicant_data.empty:
        selected_app = select_best_application(
            applicant_data, ["nb_applicants", "nb_inventors", "earliest_publn_date"]
        )
        selected_app_ids = selected_app[["docdb_family_id", "appln_id"]]
        selected_applicants = applicant_data.merge(
            selected_app_ids, on=["docdb_family_id", "appln_id"]
        )
        df_applicant_counts = (
            selected_applicants.groupby(["docdb_family_id", "person_ctry_code"])[
                "person_id"
            ]
            .nunique()
            .reset_index(name="applicant_count")
        )
    else:
        df_applicant_counts = pd.DataFrame(
            columns=["docdb_family_id", "person_ctry_code", "applicant_count"]
        )

    # Combined counts
    df_combined_counts = (
        pd.concat(
            [
                df_inventor_counts.rename(columns={"inventor_count": "combined_count"}),
                df_applicant_counts.rename(
                    columns={"applicant_count": "combined_count"}
                ),
            ],
            ignore_index=True,
        )
        .groupby(["docdb_family_id", "person_ctry_code"], as_index=False)[
            "combined_count"
        ]
        .sum()
    )

    return df_applicant_counts, df_inventor_counts, df_combined_counts


def calculate_applicants_inventors_ratios(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
    df_combined_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate applicant, inventor, and combined ratios using pre-calculated counts.
    """

    def calculate_ratio(
        df: pd.DataFrame, count_column: str, ratio_column: str
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["docdb_family_id", "person_ctry_code", ratio_column]
            )

        total_counts = (
            df.groupby("docdb_family_id")[count_column]
            .sum()
            .reset_index(name="total_count")
        )

        df = pd.merge(df, total_counts, on="docdb_family_id", how="left")
        df[ratio_column] = df[count_column] / df["total_count"]
        df[ratio_column] = df[ratio_column].fillna(0)

        return df[["docdb_family_id", "person_ctry_code", ratio_column]]

    df_applicant_ratios = calculate_ratio(
        df_applicant_counts, "applicant_count", "applicant_ratio"
    )

    df_inventor_ratios = calculate_ratio(
        df_inventor_counts, "inventor_count", "inventor_ratio"
    )

    df_combined_ratios = calculate_ratio(
        df_combined_counts, "combined_count", "combined_ratio"
    )

    return df_applicant_ratios, df_inventor_ratios, df_combined_ratios


def merge_counts_and_ratios(
    df_applicant_counts,
    df_inventor_counts,
    df_combined_counts,
    df_applicant_ratios,
    df_inventor_ratios,
    df_combined_ratios,
) -> pd.DataFrame:
    """
    Merge all count and ratio DataFrames into one combined summary.
    """
    df_final = (
        df_applicant_counts.merge(
            df_inventor_counts, on=["docdb_family_id", "person_ctry_code"], how="outer"
        )
        .merge(
            df_combined_counts, on=["docdb_family_id", "person_ctry_code"], how="outer"
        )
        .merge(
            df_applicant_ratios, on=["docdb_family_id", "person_ctry_code"], how="outer"
        )
        .merge(
            df_inventor_ratios, on=["docdb_family_id", "person_ctry_code"], how="outer"
        )
        .merge(
            df_combined_ratios, on=["docdb_family_id", "person_ctry_code"], how="outer"
        )
        .fillna(0)
    )
    return df_final
