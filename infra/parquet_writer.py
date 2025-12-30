# infra/parquet_writer.py
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterable, Tuple

def write_parquet_partitioned(
    df: pd.DataFrame,
    out_dir: str,
    partition_cols: Iterable[str] = ("symbol", "regime"),
):
    """
    Escribe Parquet particionado de forma:
    - append-safe
    - schema-estable
    - compatible con streaming / flush por chunks

    No modifica columnas ni nombres.
    """

    if df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    # ---------
    # Validación mínima
    # ---------
    for c in partition_cols:
        if c not in df.columns:
            raise ValueError(f"Partition column missing: {c}")

    # ---------
    # Normalizar dtypes (clave para estabilidad)
    # ---------
    for c in partition_cols:
        df[c] = df[c].astype(str)

    # ---------
    # Convertir a Arrow Table
    # preserve_index=False evita columnas fantasma
    # ---------
    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
        schema=None,  # dejamos que Arrow infiera UNA vez por chunk
    )

    # ---------
    # Escritura particionada (append real)
    # ---------
    pq.write_to_dataset(
        table,
        root_path=out_dir,
        partition_cols=list(partition_cols),
        compression="zstd",
        existing_data_behavior="overwrite_or_ignore",
        use_dictionary=True,
    )

