"""
Shared BigQuery download logic with checkpointing.
Skips download if data is up to date; fetches only new rows when using incremental column.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "0_data" / "raw_data" / ".checkpoints"


def _get_checkpoint_path(output_dir: Path, table_name: str) -> Path:
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{table_name}.json"


def _load_checkpoint(output_dir: Path, table_name: str) -> Optional[dict]:
    path = _get_checkpoint_path(output_dir, table_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_checkpoint(
    output_dir: Path,
    table_name: str,
    row_count: int,
    bq_count: int,
    max_incremental_value: Optional[str] = None,
) -> None:
    path = _get_checkpoint_path(output_dir, table_name)
    data = {
        "row_count": row_count,
        "bq_count_at_download": bq_count,
        "max_incremental_value": max_incremental_value,
    }
    path.write_text(json.dumps(data, indent=2))


def _get_bq_row_count(client: bigquery.Client, table_ref: str) -> int:
    result = client.query(f"SELECT COUNT(*) as n FROM `{table_ref}`").result()
    return next(iter(result)).n


def download_table(
    project_id: str,
    dataset_id: str,
    table_name: str,
    output_dir: Path,
    output_format: str = "csv",
    incremental_column: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download a BigQuery table with checkpointing.
    Skips if data is complete; fetches only new rows when incremental_column is set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_name}.{output_format}"

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_name}"

    # Get current row count in BigQuery
    logger.info("Checking BigQuery row count")
    bq_count = _get_bq_row_count(client, table_ref)
    logger.info("BigQuery has %s rows", f"{bq_count:,}")

    if not force:
        checkpoint = _load_checkpoint(output_dir, table_name)
        if checkpoint and output_path.exists():
            stored_count = checkpoint.get("row_count", 0)
            if stored_count == bq_count:
                logger.info("Data up to date (%s rows), skipping download", f"{bq_count:,}")
                return output_path

            # Try incremental fetch if we have the column and table grew
            if incremental_column and bq_count > stored_count:
                max_val = checkpoint.get("max_incremental_value")
                if max_val is not None:
                    logger.info("Fetching new rows (incremental)")
                    # Use parameterized query - STRING works for most types when compared
                    query = f"""
                        SELECT * FROM `{table_ref}`
                        WHERE CAST(`{incremental_column}` AS STRING) > @max_val
                        ORDER BY `{incremental_column}`
                    """
                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("max_val", "STRING", str(max_val))
                        ]
                    )
                    query_job = client.query(query, job_config=job_config)
                    df_new = query_job.to_dataframe(progress_bar_type="tqdm")

                    if len(df_new) == 0:
                        logger.info("No new rows found, updating checkpoint")
                        _save_checkpoint(output_dir, table_name, stored_count, bq_count, max_val)
                        return output_path

                    # Append new rows
                    if output_format == "csv":
                        df_new.to_csv(output_path, index=False, mode="a", header=False)
                    else:
                        df_existing = pd.read_parquet(output_path)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_parquet(output_path, index=False)

                    new_max = df_new[incremental_column].astype(str).max()
                    total_count = stored_count + len(df_new)
                    _save_checkpoint(
                        output_dir, table_name, total_count, bq_count, str(new_max)
                    )
                    logger.info(
                        "Appended %s rows. Total: %s",
                        f"{len(df_new):,}",
                        f"{total_count:,}",
                    )
                    return output_path

    # Full download
    logger.info("Downloading full table: %s", table_ref)
    query = f"SELECT * FROM `{table_ref}`"
    if incremental_column:
        query += f" ORDER BY `{incremental_column}`"

    query_job = client.query(query)
    df = query_job.to_dataframe(progress_bar_type="tqdm")

    logger.info("Writing %s rows to %s", f"{len(df):,}", output_path)
    with tqdm(total=1, desc="Writing file", unit="file") as pbar:
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        pbar.update(1)

    max_inc = None
    if incremental_column and incremental_column in df.columns:
        max_inc = str(df[incremental_column].astype(str).max())

    _save_checkpoint(output_dir, table_name, len(df), bq_count, max_inc)
    logger.info("Saved %s rows to %s", f"{len(df):,}", output_path)
    return output_path
