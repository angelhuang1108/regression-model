"""
Download the studies table from BigQuery and save it to 0_data/raw_data folder.
Uses checkpointing: skips if data is up to date, fetches only new rows when using --incremental-column.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bq_downloader import download_table

# Configuration - update these with your BigQuery details
PROJECT_ID = "regeneron-capstone-delta"
DATASET_ID = "regeneron_capstone_delta_dataset"
TABLE_NAME = "studies"
OUTPUT_DIR = Path(__file__).parent.parent / "0_data" / "raw_data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BigQuery studies table to 0_data/raw_data")
    parser.add_argument("--project", default=PROJECT_ID, help="GCP project ID")
    parser.add_argument("--dataset", default=DATASET_ID, help="BigQuery dataset ID")
    parser.add_argument("--table", default=TABLE_NAME, help="Table name")
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--incremental-column",
        default=None,
        help="Column for incremental fetch (e.g. updated_at, id). Fetches only new rows when table has grown.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-download, ignoring checkpoint",
    )
    args = parser.parse_args()

    download_table(
        project_id=args.project,
        dataset_id=args.dataset,
        table_name=args.table,
        output_dir=args.output_dir,
        output_format=args.format,
        incremental_column=args.incremental_column,
        force=args.force,
    )
