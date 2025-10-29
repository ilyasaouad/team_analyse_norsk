import pandas as pd
from sqlalchemy.orm import aliased

from connect_database import create_sqlalchemy_session
from models_tables import (
    TLS201_APPLN,
    TLS206_PERSON,
    TLS207_PERS_APPLN,
)

t201 = aliased(TLS201_APPLN)
t206 = aliased(TLS206_PERSON)
t207 = aliased(TLS207_PERS_APPLN)


def get_family_ids(country_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch unique DOCDB family IDs for applications where at least one associated
    person (applicant or inventor) matches the provided country code within the
    filing year range.
    """
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

        if not results:
            return pd.DataFrame({"docdb_family_id": pd.Series(dtype="int64")})

        df_family_ids = pd.DataFrame(
            results, columns=["appln_id", "docdb_family_id", "appln_filing_year"]
        )
        df_unique_family_ids = df_family_ids["docdb_family_id"].drop_duplicates()

        return pd.DataFrame(df_unique_family_ids, columns=["docdb_family_id"])
