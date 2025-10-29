# python main.py -c NO -s 2022 -e 2023 -r 10

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# Configure pandas display options for CLI
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.width', None)        # Auto-detect width
pd.set_option('display.max_colwidth', None) # Show full column content

# Our functions
from get_applicants_inventors_details import (
    get_applicant_inventor,
    get_family_ids,
)
from config import Config

# ---------------------------
# Logging
# ---------------------------
def setup_logging(log_dir: Path = None) -> logging.Logger:
    """Configure logging for both console and file output."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-run interactively
    if logger.handlers:
        return logger

    # --- Console handler (for CLI output) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File handler (for persistent logs) ---
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


def create_data_folder(country_code: str, start_year: int, end_year: int, working_dir: Path) -> Path:
    """
    Create a folder for storing data and return the output directory path.
    Example: DataTables_NO_2020_2021
    """
    folder_name = f"DataTables_{country_code}_{start_year}_{end_year}"
    output_dir = working_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_dfs_to_csv(dfs_tuple, df_names, csv_output_dir: Path):
    """
    Save DataFrames from a tuple to separate CSV files.
    """
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for df_item, name in zip(dfs_tuple, df_names):
        filepath = csv_output_dir / f"{name}.csv"
        if isinstance(df_item, pd.DataFrame):
            if df_item.empty:
                logger.warning(f"DataFrame '{name}' is empty. Creating empty file: {filepath}")
            df_item.to_csv(filepath, index=False)
            logger.info(f"Saved DataFrame '{name}' to {filepath}")
            written_files.append(filepath)
        else:
            # This part of the original function is no longer needed for this use case
            # but is kept for general utility.
            value_df = pd.DataFrame({"value": [df_item]})
            value_df.to_csv(filepath, index=False)
            logger.info(f"Saved value '{name}' to {filepath}")
            written_files.append(filepath)
            
    return written_files


# ---------------------------
# CLI Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Patent Data Analysis (CLI). Fetch and save applicant/inventor tables to CSV."
    )
    parser.add_argument(
        "--country-code", "-c",
        default="NO",
        help="Country code (default: NO)"
    )
    parser.add_argument(
        "--start-year", "-s",
        type=int, default=2020,
        help="Start year (default: 2020)"
    )
    parser.add_argument(
        "--end-year", "-e",
        type=int, default=2020,
        help="End year (default: 2020)"
    )
    parser.add_argument(
        "--working-dir", "-w",
        type=Path, default=Path.cwd(),
        help="Working directory where outputs will be written (default: current directory)"
    )
    parser.add_argument(
        "--range-limit", "-r",
        type=int, default=None, # Changed default to None for full runs
        help="Optional limit on the number of family IDs to process for testing. If not set, processes all."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    country_code = args.country_code.upper()
    start_year = args.start_year
    end_year = args.end_year
    working_dir = normalize_working_dir(args.working_dir)
    range_limit = args.range_limit

    # Basic sanity checks
    if start_year > end_year:
        logger.error("Start year cannot be greater than end year.")
        sys.exit(1)

    try:
        logger.info(
            f"Starting data processing for country: {country_code}, years: {start_year}-{end_year}"
        )
        if range_limit:
            logger.info(f"Applying a range limit of {range_limit} family IDs for testing.")

        # Create output directory
        output_dir = create_data_folder(country_code, start_year, end_year, working_dir)

        # Update Config with new settings
        Config.update(
            output_dir=str(output_dir),
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
        )

        # Fetch family IDs
        df_unique_family_ids = get_family_ids(
            Config.country_code, Config.start_year, Config.end_year
        )

        if df_unique_family_ids.empty:
            logger.warning("No family IDs found for the given criteria.")
            print("\n=== DataFrame: Unique Family IDs ===")
            print(df_unique_family_ids)
            print(f"\nShape: {df_unique_family_ids.shape}")
            print("\nNo data to export. Done.")
            return

        logger.info(f"Fetched {len(df_unique_family_ids)} unique family IDs.")

        if range_limit:
            df_unique_family_ids = df_unique_family_ids.head(range_limit)
            logger.info(f"Applying range limit of {len(df_unique_family_ids)} family IDs after fetch.")

        # --- REFACTORED: Names aligned with the two DataFrames ---
        df_names = ["unique_family_ids"]

        # Display DataFrames
        print("\n=== DataFrame: Unique Family IDs ===")
        print(df_unique_family_ids)
        print(f"\nShape: {df_unique_family_ids.shape}")

        # Save DataFrames to CSV
        csv_output_dir = Path(Config.output_dir) / "data" / "applicants_inventors"
        written_files = save_dfs_to_csv((df_unique_family_ids,), df_names, csv_output_dir)

        family_ids_list = df_unique_family_ids["docdb_family_id"].tolist()
        df_applicant_inventor = get_applicant_inventor(family_ids_list)
        details_csv_path = (
            Path(Config.output_dir)
            / "data"
            / "applicants_inventors"
            / "applicant_inventor_details.csv"
        )

        # --- REFACTORED: Enhanced console summary ---
        print("\n=== Patent Data Export Summary ===")
        print(f"Country:      {country_code}")
        print(f"Years:        {start_year}-{end_year}")
        print(f"Family IDs:   {len(df_unique_family_ids)}")
        print(f"Details Recs: {len(df_applicant_inventor)}")
        print(f"Output Dir:   {csv_output_dir}")
        print("\nFiles written:")
        for p in written_files:
            print(f" - {p.name}")
        print(f" - {details_csv_path.name}")
        print("\nDone.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
