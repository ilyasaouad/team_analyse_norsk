# python main.py -c NO -s 2022 -e 2023 -r 10
# https://teamanalysenors-fuzo53p4bkgpvizgrxwjdl.streamlit.app/
# Passord: team-analyse

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# Configure pandas display options for CLI
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Import main data functions
from get_applicants_inventors_details import (
    get_applicant_inventor,
    get_family_ids,
    calculate_applicants_inventors_counts,
    calculate_applicants_inventors_ratios,
)
from config import Config


# ---------------------------
# Logging
# ---------------------------
def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configure logging for both console and file output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


logger = setup_logging()


# ---------------------------
# Helpers
# ---------------------------
def normalize_working_dir(path_value) -> Path:
    """Normalize working directory, supporting Git-Bash style paths like /c/Users/..."""
    path_str = str(path_value).strip()
    if not path_str:
        return Path.cwd()

    converted = False
    if (
        (path_str.startswith("/") or path_str.startswith("\\"))
        and len(path_str) > 2
        and path_str[1].isalpha()
        and path_str[2] in ("/", "\\")
    ):
        drive = f"{path_str[1].upper()}:"
        remainder = path_str[2:]
        path_str = f"{drive}{remainder}"
        converted = True

    normalized = Path(path_str).expanduser()
    try:
        normalized = normalized.resolve()
    except FileNotFoundError:
        normalized = normalized.absolute()

    if converted:
        logger.info(f"Normalized working directory to {normalized}")

    return normalized


def create_data_folder(
    country_code: str, start_year: int, end_year: int, working_dir: Path
) -> Path:
    folder_name = f"DataTables_{country_code}_{start_year}_{end_year}"
    output_dir = working_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_dfs_to_csv(dfs_tuple, df_names, csv_output_dir: Path):
    """Save multiple DataFrames to CSV files."""
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    for df_item, name in zip(dfs_tuple, df_names):
        filepath = csv_output_dir / f"{name}.csv"
        if isinstance(df_item, pd.DataFrame):
            if df_item.empty:
                logger.warning(
                    f"DataFrame '{name}' is empty. Creating empty file: {filepath}"
                )
            df_item.to_csv(filepath, index=False)
            logger.info(f"Saved DataFrame '{name}' to {filepath}")
            written_files.append(filepath)
        else:
            value_df = pd.DataFrame({"value": [df_item]})
            value_df.to_csv(filepath, index=False)
            logger.info(f"Saved value '{name}' to {filepath}")
            written_files.append(filepath)
    return written_files


def merge_counts_and_ratios(
    df_applicant_counts: pd.DataFrame,
    df_inventor_counts: pd.DataFrame,
    df_combined_counts: pd.DataFrame,
    df_applicant_ratios: pd.DataFrame,
    df_inventor_ratios: pd.DataFrame,
    df_combined_ratios: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all count and ratio DataFrames into one combined summary DataFrame.
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


# ---------------------------
# CLI Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Patent Data Analysis (CLI). Fetch, analyze, and export applicant/inventor data."
    )
    parser.add_argument(
        "--country-code", "-c", default="NO", help="Country code (default: NO)"
    )
    parser.add_argument(
        "--start-year", "-s", type=int, default=2020, help="Start year (default: 2020)"
    )
    parser.add_argument(
        "--end-year", "-e", type=int, default=2020, help="End year (default: 2020)"
    )
    parser.add_argument(
        "--working-dir",
        "-w",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: current)",
    )
    parser.add_argument(
        "--range-limit",
        "-r",
        type=int,
        default=None,
        help="Optional family ID limit for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    country_code = args.country_code.upper()
    start_year = args.start_year
    end_year = args.end_year
    working_dir = normalize_working_dir(args.working_dir)
    range_limit = args.range_limit

    if start_year > end_year:
        logger.error("Start year cannot be greater than end year.")
        sys.exit(1)

    try:
        logger.info(
            f"Starting data processing for {country_code} ({start_year}-{end_year})"
        )

        # Step 1: Prepare output directory
        output_dir = create_data_folder(country_code, start_year, end_year, working_dir)
        Config.update(
            output_dir=str(output_dir),
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
        )

        # Step 2: Fetch unique family IDs
        df_unique_family_ids = get_family_ids(country_code, start_year, end_year)
        if df_unique_family_ids.empty:
            logger.warning("No family IDs found for given parameters.")
            print(df_unique_family_ids)
            return

        if range_limit:
            df_unique_family_ids = df_unique_family_ids.head(range_limit)
            logger.info(
                f"Using limited dataset ({len(df_unique_family_ids)} family IDs)."
            )

        family_ids_list = df_unique_family_ids["docdb_family_id"].tolist()
        
        # Save unique family IDs
        family_ids_dir = Path(Config.output_dir) / "data" / "applicants_inventors"
        family_ids_dir.mkdir(parents=True, exist_ok=True)
        family_ids_path = family_ids_dir / "unique_family_ids.csv"
        df_unique_family_ids.to_csv(family_ids_path, index=False)
        logger.info(f"Saved unique family IDs to: {family_ids_path}")

        # Step 3: Fetch applicant/inventor details
        df_applicant_inventor = get_applicant_inventor(family_ids_list)

        # Step 4: Calculate counts and ratios
        logger.info("Calculating applicant/inventor counts and ratios...")
        df_applicant_counts, df_inventor_counts, df_combined_counts = (
            calculate_applicants_inventors_counts(df_applicant_inventor)
        )

        df_applicant_ratios, df_inventor_ratios, df_combined_ratios = (
            calculate_applicants_inventors_ratios(
                df_applicant_counts, df_inventor_counts, df_combined_counts
            )
        )

        # Step 5: Save all DataFrames separately
        csv_output_dir = Path(Config.output_dir) / "data" / "analysis"
        df_names = [
            "applicant_counts",
            "inventor_counts",
            "combined_counts",
            "applicant_ratios",
            "inventor_ratios",
            "combined_ratios",
        ]
        dfs_tuple = (
            df_applicant_counts,
            df_inventor_counts,
            df_combined_counts,
            df_applicant_ratios,
            df_inventor_ratios,
            df_combined_ratios,
        )
        written_files = save_dfs_to_csv(dfs_tuple, df_names, csv_output_dir)

        # Step 6: Merge all count and ratio DataFrames
        logger.info("Merging all count and ratio DataFrames...")
        df_summary = merge_counts_and_ratios(
            df_applicant_counts,
            df_inventor_counts,
            df_combined_counts,
            df_applicant_ratios,
            df_inventor_ratios,
            df_combined_ratios,
        )
        summary_path = csv_output_dir / "counts_ratios_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        written_files.append(summary_path)
        logger.info(f"Saved merged counts+ratios summary to: {summary_path}")

        # Step 7: CLI summary
        print("\n=== Patent Data Export Summary ===")
        print(f"Country:      {country_code}")
        print(f"Years:        {start_year}-{end_year}")
        print(f"Families:     {len(df_unique_family_ids)}")
        print(f"Records:      {len(df_applicant_inventor)}")
        print(f"Output Dir:   {csv_output_dir}")
        print("\nFiles written:")
        for p in written_files:
            print(f" - {Path(p).name}")
        print("\nDone.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
