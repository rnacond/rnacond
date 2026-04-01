import decoupler as dc  # type: ignore
import logging
import anndata as ad
logger = logging.getLogger(__name__)
from typing import Union, Optional

def create_pseudobulk(
    adata: Union[str, ad.AnnData],
    sample_col: str,
    groups_col: Optional[str] = None,
    layer: Optional[str] = None,
    min_cells: int = 0,
    min_counts: int = 0,
    mode: str = "sum",
    remove_empty: bool = False
) -> ad.AnnData:
    """Create pseudobulk profiles from single-cell data.

    Args:
        adata (Union[str, ad.AnnData]): Input AnnData object or path to h5ad file.
        sample_col (str): Column in adata.obs containing sample identifiers.
        groups_col (Optional[str], optional): Column in adata.obs containing group identifiers (e.g., cell types). Defaults to None.
        layer (Optional[str], optional): Layer of adata to use for pseudobulk. If None, use adata.X. Defaults to None.
        min_cells (int, optional): Minimum number of cells required per group. Defaults to 10.
        min_counts (int, optional): Minimum number of counts required per group. Defaults to 500.
        mode (str, optional): How to aggregate the data ('sum' or 'mean'). Defaults to "sum".
        remove_empty (bool, optional): Whether to remove groups with no cells after filtering. Defaults to True.

    Returns:
        anndata.AnnData: AnnData object containing pseudobulk profiles.

    Raises:
        ValueError: If required columns are missing from adata.obs.
        Exception: If an error occurs during pseudobulk creation.
    """
    try:
        print('Pseudo bulking...')
        adata = _load_adata(adata)
        _validate_columns(adata, sample_col, groups_col)
        pseudobulk = dc.get_pseudobulk(
            adata,
            sample_col=sample_col,
            groups_col=groups_col,
            layer=layer,
            mode=mode,
            min_cells=min_cells,
            min_counts=min_counts,
            remove_empty=remove_empty,
        )
        _log_pseudobulk_summary(pseudobulk)
        return pseudobulk
    except Exception as e:
        logger.error(f"Error creating pseudobulk profiles: {e}")
        raise
    
def _load_adata(adata: Union[str, ad.AnnData]) -> ad.AnnData:
    """Load AnnData object from a file path if necessary.

    Args:
        adata (Union[str, ad.AnnData]): Input AnnData object or path to h5ad file.

    Returns:
        anndata.AnnData: Loaded AnnData object.
    """
    if isinstance(adata, str):
        return ad.read_h5ad(adata)
    return adata


def _validate_columns(
    adata: ad.AnnData, sample_col: str, groups_col: Optional[str]
) -> None:
    """Validate the presence of required columns in adata.obs.

    Args:
        adata (ad.AnnData): The AnnData object to validate.
        sample_col (str): The sample column name.
        groups_col (Optional[str]): The groups column name.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {sample_col}
    if groups_col:
        required_cols.add(groups_col)
    missing_cols = required_cols - set(adata.obs.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")


def _log_pseudobulk_summary(pseudobulk: ad.AnnData) -> None:
    """Log summary statistics of the pseudobulk data.

    Args:
        pseudobulk (ad.AnnData): The pseudobulk AnnData object.
    """
    logger.info("Created pseudobulk profiles:")
    logger.info(f"- Number of samples: {pseudobulk.n_obs}")
    logger.info(f"- Number of genes: {pseudobulk.n_vars}")