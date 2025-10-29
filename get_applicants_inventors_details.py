import logging
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

        # Using batch for long dataset
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
