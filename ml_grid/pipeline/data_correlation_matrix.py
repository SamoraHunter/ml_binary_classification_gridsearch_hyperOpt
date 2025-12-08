import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("ml_grid")


def handle_correlation_matrix(
    local_param_dict: Dict,
    drop_list: List[str],
    df: pd.DataFrame,
    chunk_size: int = 1000,
) -> List[str]:
    """
    Hybrid Correlation Optimizer.

    Features:
    1. Respects existing 'drop_list' (adds to it, doesn't replace it).
    2. Optimizes by skipping columns already in 'drop_list'.
    3. Hybrid GPU/CPU execution with robust error handling.
    """

    threshold = local_param_dict.get("corr", 0.25)

    # Filter to numeric columns only
    numeric_columns = df.select_dtypes(include=["number"]).columns

    if len(numeric_columns) == 0:
        return drop_list  # Return existing list if no new work to do

    logger.info("Preparing data (converting to float32)...")
    df_numeric = df[numeric_columns]
    col_names = df_numeric.columns.tolist()

    # Create a mapping for fast index lookups
    col_to_idx = {name: i for i, name in enumerate(col_names)}

    # Convert data to float32
    data = df_numeric.values.astype(np.float32)

    # --- GPU DETECTION & SAFETY ---
    use_gpu = False
    try:
        import cupy as cp

        if cp.cuda.is_available():
            free_mem = cp.cuda.Device().mem_info[0]
            req_mem = (data.shape[1] ** 2) * 4  # 4 bytes per float32

            if free_mem > req_mem * 1.2:
                use_gpu = True
                logger.info(
                    f"GPU Detected: {cp.cuda.Device().name}. Free VRAM: {free_mem/1e9:.2f} GB."
                )
            else:
                logger.warning(
                    "GPU detected but insufficient VRAM. Falling back to CPU."
                )
    except Exception as e:
        logger.warning(f"GPU acceleration unavailable (falling back to CPU): {e}")
        use_gpu = False
    # -----------------------------

    # Convert input drop_list to a Set for O(1) lookups
    existing_drops = set(drop_list)

    if use_gpu:
        try:
            return _process_on_gpu(data, col_names, threshold, existing_drops)
        except Exception as e:
            logger.error(f"GPU processing failed: {e}. Retrying on CPU.")
            # Fallthrough to CPU
            pass

    # CPU Fallback
    return _process_on_cpu(
        data, col_names, col_to_idx, threshold, chunk_size, existing_drops
    )


def _process_on_gpu(
    data: np.ndarray, col_names: List[str], threshold: float, existing_drops: Set[str]
) -> List[str]:
    import cupy as cp

    n_samples = data.shape[0]

    # Initialize the final set with what we already had
    to_drop = existing_drops.copy()

    # Move data to GPU
    gpu_data = cp.asarray(data)

    # Standardize
    means = gpu_data.mean(axis=0, keepdims=True)
    stds = gpu_data.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    gpu_data = (gpu_data - means) / stds

    scale_factor = 1.0 / (n_samples - 1)

    # Matrix Multiplication
    corr_matrix = cp.matmul(gpu_data.T, gpu_data)
    corr_matrix *= scale_factor
    corr_matrix = cp.abs(corr_matrix)

    # Upper Triangle only (k=1)
    upper_tri = cp.triu(corr_matrix, k=1)

    # Get indices of high correlations
    rows, cols = cp.where(upper_tri > threshold)

    cpu_rows = cp.asnumpy(rows)
    cpu_cols = cp.asnumpy(cols)

    # Process pairs
    for i, j in zip(cpu_rows, cpu_cols):
        col_i = col_names[i]
        col_j = col_names[j]

        # KEY LOGIC: If Col_I is already marked for drop (either from input list
        # or from this loop), we skip. Otherwise, we drop Col_J.
        if col_i not in to_drop:
            to_drop.add(col_j)

    logger.info(f"GPU complete. Total columns to drop: {len(to_drop)}")
    return sorted(list(to_drop))


def _process_on_cpu(
    data: np.ndarray,
    col_names: List[str],
    col_to_idx: Dict[str, int],
    threshold: float,
    chunk_size: int,
    existing_drops: Set[str],
) -> List[str]:

    logger.info("Using optimized CPU processing...")
    n_samples, n_cols = data.shape

    # Standardize
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    data = (data - means) / stds

    scale_factor = 1.0 / (n_samples - 1)

    # Initialize mask with PRE-EXISTING drops
    # This optimizes the loop: we won't calculate correlations for columns
    # that came in already dropped.
    dropped_mask = np.zeros(n_cols, dtype=bool)

    for col in existing_drops:
        if col in col_to_idx:
            dropped_mask[col_to_idx[col]] = True

    effective_chunk_size = max(chunk_size, 500)

    with tqdm(total=n_cols, desc="CPU Correlation") as pbar:
        for i in range(0, n_cols, effective_chunk_size):
            i_end = min(i + effective_chunk_size, n_cols)

            chunk_data = data[:, i:i_end]

            # Correlation Block
            corr_chunk = np.matmul(chunk_data.T, data) * scale_factor
            corr_chunk = np.abs(corr_chunk)

            for local_row in range(corr_chunk.shape[0]):
                global_current_idx = i + local_row

                # OPTIMIZATION:
                # If this column was in the input drop_list OR we just dropped it, SKIP.
                if dropped_mask[global_current_idx]:
                    continue

                # Check neighbors to the right
                candidates = corr_chunk[local_row, global_current_idx + 1 :]
                hits = np.where(candidates > threshold)[0]

                if hits.size > 0:
                    # Add to mask
                    dropped_mask[global_current_idx + 1 + hits] = True

            pbar.update(i_end - i)

    # Convert mask back to list
    dropped_indices = np.where(dropped_mask)[0]
    newly_identified_drops = {col_names[i] for i in dropped_indices}

    # Merge with original list (in case original list had cols not in this dataframe)
    final_drop_set = existing_drops.union(newly_identified_drops)

    logger.info(f"CPU complete. Total columns to drop: {len(final_drop_set)}")
    return sorted(list(final_drop_set))
