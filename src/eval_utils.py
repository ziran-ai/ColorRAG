from math import inf
import os
import logging
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.sparse.csr import spmatrix
from scipy.stats import chi2
from typing import Mapping, Sequence, Tuple, Iterable, Union
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

import psutil
import scib


_cpu_count: Union[None, int] = psutil.cpu_count(logical=False)
if _cpu_count is None:
    _cpu_count: int = psutil.cpu_count(logical=True)
_logger = logging.getLogger(__name__)


def evaluate(adata: ad.AnnData,
             n_epoch: int,
             embedding_key: str = 'delta',
             n_neighbors: int = 15,
             resolutions: Iterable[float] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
             clustering_method: str = "leiden",
             cell_type_col: str = "cell_type",
             batch_col: Union[str, None] = "batch_indices",
             color_by: Iterable[str] = None,
             return_fig: bool = False,
             plot_fname: str = "umap",
             plot_ftype: str = "jpg",
             plot_dir: Union[str, None] = None,
             plot_dpi: int = 300,
             min_dist: float = 0.3,
             spread: float = 1,
             n_jobs: int = 1,
             random_state: Union[None, int, np.random.RandomState, np.random.Generator] = 0,
             umap_kwargs: dict = dict()
             ) -> Mapping[str, Union[float, None, Figure]]:
    """Evaluates the clustering and batch correction performance of the given
    embeddings, and optionally plots the embeddings.

    Embeddings will be plotted if return_fig is True or plot_dir is provided.
    When tensorboard_dir is provided, will also save the embeddings using a
    tensorboard SummaryWriter.

    NOTE: Set n_jobs to 1 if you encounter pickling error.

    Args:
        adata: the dataset with the embedding to be evaluated.
        embedding_key: the key to the embedding. Must be in adata.obsm.
        n_neighbors: #neighbors used when computing neithborhood graph and
            calculating entropy of batch mixing / kBET.
        resolutions: a sequence of resolutions used for clustering.
        clustering_method: clustering method used. Should be one of 'leiden' or
            'louvain'.
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot.
        color_by: a list of adata.obs column keys to color the embeddings by.
            If None, will look up adata.uns['color_by']. Only used if is
            drawing.
        plot_fname: file name of the generated plot. Only used if is drawing.
        plot_ftype: file type of the generated plot. Only used if is drawing.
        plot_dir: directory to save the generated plot. If None, do not save
            the plot.
        plot_dpi: dpi to save the plot.
        writer: an initialized SummaryWriter to save the UMAP plot to. Only
            used if is drawing.
        min_dist: the min_dist argument in sc.tl.umap. Only used is drawing.
        spread: the spread argument in sc.tl.umap. Only used if is drawing.
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        random_state: random state for knn calculation.
        umap_kwargs: other kwargs to pass to sc.pl.umap.

    Returns:
        A dict storing the ari, nmi, asw, ebm, k_bet of the cell embeddings
        with key "ari", "nmi", "asw", "ebm", "k_bet", respectively. If draw is
        True and return_fig is True, will also store the plotted figure with
        key "fig".
    """

    if cell_type_col and not pd.api.types.is_categorical_dtype(adata.obs[cell_type_col]):
        #_logger.warning("scETM.evaluate assumes discrete cell types. Converting cell_type_col to categorical.")
        adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).astype('category')
    if batch_col and not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        #_logger.warning("scETM.evaluate assumes discrete batches. Converting batch_col to categorical.")
        adata.obs[batch_col] = adata.obs[batch_col].astype(str).astype('category')

    # calculate neighbors
    _get_knn_indices(adata, use_rep=embedding_key, n_neighbors=n_neighbors, random_state=random_state, calc_knn=True)

    # calculate clustering metrics
    if cell_type_col in adata.obs and len(resolutions) > 0:
        cluster_key, best_ari, best_nmi = clustering(adata, resolutions=resolutions, cell_type_col=cell_type_col, batch_col=batch_col, clustering_method=clustering_method)
    else:
        cluster_key = best_ari = best_nmi = None

    if adata.obs[cell_type_col].nunique() > 1:
        sw = silhouette_samples(adata.X if embedding_key == 'X' else adata.obsm[embedding_key],
                                adata.obs[cell_type_col])
        adata.obs['silhouette_width'] = sw
        asw = np.mean(sw)
        #print(f'{embedding_key}_ASW: {asw:7.4f}')

        asw_2 = scib.me.silhouette(adata, label_key=cell_type_col, embed=embedding_key)


        if batch_col and cell_type_col:
            sw_table = adata.obs.pivot_table(index=cell_type_col, columns=batch_col, values="silhouette_width",
                                             aggfunc="mean")
            #print(f'SW: {sw_table}')
            if plot_dir is not None:
                sw_table.to_csv(os.path.join(plot_dir, f'{plot_fname}.csv'))
    else:
        asw = 0.
        asw_2 = 0.

    # calculate batch correction metrics
    need_batch = batch_col and adata.obs[batch_col].nunique() > 1
    if need_batch:
        ebm = calculate_entropy_batch_mixing(adata,
                                             use_rep=embedding_key,
                                             batch_col=batch_col,
                                             n_neighbors=n_neighbors,
                                             calc_knn=False,
                                             n_jobs=n_jobs,
                                             )
        #print(f'{embedding_key}_BE: {ebm:7.4f}')
        k_bet = calculate_kbet(adata,
                               use_rep=embedding_key,
                               batch_col=batch_col,
                               n_neighbors=n_neighbors,
                               calc_knn=False,
                               n_jobs=n_jobs,
                               )[2]
        #print(f'{embedding_key}_kBET: {k_bet:7.4f}')
        batch_asw = scib.me.silhouette_batch(adata, batch_key=batch_col, label_key=cell_type_col, embed=embedding_key, verbose=False)
        batch_graph_score = get_graph_connectivity(adata, use_rep=embedding_key,)
    else:
        ebm = k_bet = batch_asw = batch_graph_score = None

    # plot UMAP embeddings
    if return_fig or plot_dir is not None:
        if color_by is None:
            color_by = [batch_col, cell_type_col] if need_batch else [cell_type_col]
        color_by = list(color_by)
        if 'color_by' in adata.uns:
            for col in adata.uns['color_by']:
                if col not in color_by:
                    color_by.insert(0, col)
        if cluster_key is not None:
            color_by = [cluster_key] + color_by
        fig = draw_embeddings(adata=adata, color_by=color_by,
                              min_dist=min_dist, spread=spread,
                              ckpt_dir=plot_dir, fname=f'{plot_fname+str(n_epoch)}.{plot_ftype}', return_fig=return_fig,
                              dpi=plot_dpi,
                              umap_kwargs=umap_kwargs)
    else:
        fig = None

    return dict(
        ari=best_ari,
        nmi=best_nmi,
        asw=asw,
        asw_2=asw_2,
        ebm=ebm,
        k_bet=k_bet,
        batch_asw=batch_asw,
        batch_graph_score=batch_graph_score,
        fig=fig
    )

def evaluate_ari(adata: ad.AnnData,
             n_epoch: int,
             embedding_key: str = 'delta',
             n_neighbors: int = 15,
             resolutions: Iterable[float] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
             clustering_method: str = "leiden",
             cell_type_col: str = "cell_type",
             batch_col: Union[str, None] = "batch_indices",
             color_by: Iterable[str] = None,
             return_fig: bool = False,
             plot_fname: str = "umap",
             plot_ftype: str = "jpg",
             plot_dir: Union[str, None] = None,
             plot_dpi: int = 300,
             min_dist: float = 0.3,
             spread: float = 1,
             n_jobs: int = 1,
             random_state: Union[None, int, np.random.RandomState, np.random.Generator] = 0,
             umap_kwargs: dict = dict()
             ) -> Mapping[str, Union[float, None, Figure]]:
    """Evaluates the clustering and batch correction performance of the given
    embeddings, and optionally plots the embeddings.

    Embeddings will be plotted if return_fig is True or plot_dir is provided.
    When tensorboard_dir is provided, will also save the embeddings using a
    tensorboard SummaryWriter.

    NOTE: Set n_jobs to 1 if you encounter pickling error.

    Args:
        adata: the dataset with the embedding to be evaluated.
        embedding_key: the key to the embedding. Must be in adata.obsm.
        n_neighbors: #neighbors used when computing neithborhood graph and
            calculating entropy of batch mixing / kBET.
        resolutions: a sequence of resolutions used for clustering.
        clustering_method: clustering method used. Should be one of 'leiden' or
            'louvain'.
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot.
        color_by: a list of adata.obs column keys to color the embeddings by.
            If None, will look up adata.uns['color_by']. Only used if is
            drawing.
        plot_fname: file name of the generated plot. Only used if is drawing.
        plot_ftype: file type of the generated plot. Only used if is drawing.
        plot_dir: directory to save the generated plot. If None, do not save
            the plot.
        plot_dpi: dpi to save the plot.
        writer: an initialized SummaryWriter to save the UMAP plot to. Only
            used if is drawing.
        min_dist: the min_dist argument in sc.tl.umap. Only used is drawing.
        spread: the spread argument in sc.tl.umap. Only used if is drawing.
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        random_state: random state for knn calculation.
        umap_kwargs: other kwargs to pass to sc.pl.umap.

    Returns:
        A dict storing the ari, nmi, asw, ebm and k_bet of the cell embeddings
        with key "ari", "nmi", "asw", "ebm", "k_bet", respectively. If draw is
        True and return_fig is True, will also store the plotted figure with
        key "fig".
    """

    if cell_type_col and not pd.api.types.is_categorical_dtype(adata.obs[cell_type_col]):
        #_logger.warning("scETM.evaluate assumes discrete cell types. Converting cell_type_col to categorical.")
        adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).astype('category')
    if batch_col and not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        #_logger.warning("scETM.evaluate assumes discrete batches. Converting batch_col to categorical.")
        adata.obs[batch_col] = adata.obs[batch_col].astype(str).astype('category')

    # calculate neighbors
    _get_knn_indices(adata, use_rep=embedding_key, n_neighbors=n_neighbors, random_state=random_state, calc_knn=True)

    # calculate clustering metrics
    if cell_type_col in adata.obs and len(resolutions) > 0:
        cluster_key, best_ari, best_nmi = clustering(adata, resolutions=resolutions, cell_type_col=cell_type_col, batch_col=batch_col, clustering_method=clustering_method)
    else:
        cluster_key = best_ari = best_nmi = None

    return best_ari

def _eff_n_jobs(n_jobs: Union[None, int]) -> int:
    """If n_jobs <= 0, set it as the number of physical cores _cpu_count"""
    if n_jobs is None:
        return 1
    return int(n_jobs) if n_jobs > 0 else _cpu_count


def _calculate_kbet_for_one_chunk(knn_indices, attr_values, ideal_dist, n_neighbors):
    dof = ideal_dist.size - 1

    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    for i in range(ns):
        # NOTE: Do not use np.unique. Some of the batches may not be present in
        # the neighborhood.
        observed_counts = pd.Series(attr_values[knn_indices[i, :]]).value_counts(sort=False).values
        expected_counts = ideal_dist * n_neighbors
        stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


# def _get_knn_indices(adata: ad.AnnData,
#                      use_rep: str = "delta",
#                      n_neighbors: int = 25,
#                      random_state: int = 0,
#                      calc_knn: bool = True
#                      ) -> np.ndarray:
#     if calc_knn:
#         assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
#         neighbors = sc.Neighbors(adata)
#         neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state,write_knn_indices=True)
#         #neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state)
#         #adata.obsm['knn_indices'] = neighbors.knn_indices
#         adata.obsp['distances'] = neighbors.distances
#         adata.obsp['connectivities'] = neighbors.connectivities
#         adata.obsm['knn_indices'] = neighbors.knn_indices
#         adata.uns['neighbors'] = {
#             'connectivities_key': 'connectivities',
#             'distances_key': 'distances',
#             'knn_indices_key': 'knn_indices',
#             'params': {
#                 'n_neighbors': n_neighbors,
#                 'use_rep': use_rep,
#                 'metric': 'euclidean',
#                 'method': 'umap'
#             }
#         }
#     else:
#         assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
#         assert adata.uns['neighbors']['params'][
#                    'n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"

#     return adata.obsm['knn_indices']
def _get_knn_indices(
    adata: ad.AnnData,
    use_rep: str = "delta",
    n_neighbors: int = 25,
    random_state: int = 0,
    calc_knn: bool = True
) -> np.ndarray:
    """Get KNN indices from AnnData object with robust shape handling.
    
    Args:
        adata: AnnData object containing the data
        use_rep: Key in adata.obsm where the embedding is stored
        n_neighbors: Number of neighbors to find
        random_state: Random seed for reproducibility
        calc_knn: Whether to recompute KNN if not already present
        
    Returns:
        Array containing neighbor indices with shape (n_cells, actual_neighbors)
    """
    if calc_knn:
        assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
        
        # Compute neighbors with safety checks
        if n_neighbors >= adata.shape[0]:
            n_neighbors = max(1, adata.shape[0] - 1)
            _logger.warning(f"Reduced n_neighbors to {n_neighbors} (number of cells: {adata.shape[0]})")
            
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            random_state=random_state,
            method='umap'
        )
        
        # Convert sparse distance matrix to knn_indices with dynamic neighbor count
        distances = adata.obsp['distances']
        n_cells = adata.shape[0]
        
        # First pass to determine max actual neighbors
        max_actual_neighbors = 0
        neighbor_counts = []
        for i in range(n_cells):
            neighbors = distances[i].nonzero()[1]
            neighbor_counts.append(len(neighbors))
            if len(neighbors) > max_actual_neighbors:
                max_actual_neighbors = len(neighbors)
        
        # Create array with actual max neighbors
        knn_indices = np.full((n_cells, max_actual_neighbors), -1, dtype=int)
        for i in range(n_cells):
            neighbors = distances[i].nonzero()[1]
            knn_indices[i, :len(neighbors)] = neighbors
            
        # Store results
        adata.obsm['knn_indices'] = knn_indices
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'knn_indices_key': 'knn_indices',
            'params': {
                'n_neighbors': n_neighbors,
                'use_rep': use_rep,
                'metric': 'euclidean',
                'method': 'umap'
            }
        }
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        knn_indices = adata.obsm['knn_indices']
    
    return knn_indices
"""
def _get_knn_indices(adata: ad.AnnData,
                     use_rep: str = "delta",
                     n_neighbors: int = 25,
                     random_state: int = 0,
                     calc_knn: bool = True
                     ) -> np.ndarray:
    if calc_knn:
        assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state)
        adata.obsp['distances'] = neighbors.distances
        adata.obsp['connectivities'] = neighbors.connectivities
        
        # 尝试从 adata.uns['neighbors'] 中获取 indices
        if 'neighbors' in adata.uns and 'indices' in adata.uns['neighbors']:
            knn_indices = adata.uns['neighbors']['indices']
        else:
            # 如果没有直接的 indices，可能需要重新计算或从其他属性中获取
            raise ValueError("Failed to retrieve indices from adata.uns['neighbors'].")
        
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {
                'n_neighbors': n_neighbors,
                'use_rep': use_rep,
                'metric': 'euclidean',
                'method': 'umap'
            }
        }
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        assert adata.uns['neighbors']['params'][
                   'n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"
        knn_indices = adata.uns['neighbors']['indices']

    return knn_indices
"""
def get_graph_connectivity(
        adata: ad.AnnData,
        use_rep: str = "delta",):

    sc.pp.neighbors(adata, use_rep=use_rep)
    score = scib.me.graph_connectivity(adata, label_key='cell_types')
    return score


def _calculate_kbet_for_one_chunk(knn_indices, attr_values, ideal_dist, n_neighbors):
    """Calculate kBET statistics for a chunk of cells."""
    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    
    for i in range(ns):
        # Get batch labels for neighbors
        neighbor_batches = attr_values[knn_indices[i, :]]
        
        # Get unique batches present in neighborhood
        present_batches = np.unique(neighbor_batches)
        n_present = len(present_batches)
        
        # Calculate observed and expected counts only for present batches
        observed_counts = np.bincount(neighbor_batches, minlength=len(ideal_dist))[present_batches]
        expected_counts = ideal_dist[present_batches] * n_neighbors
        
        # Calculate chi-square statistic
        stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_value = 1 - chi2.cdf(stat, n_present - 1)
        
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


def calculate_kbet(
        adata: ad.AnnData,
        use_rep: str = "delta",
        batch_col: str = "batch_indices",
        n_neighbors: int = 25,
        alpha: float = 0.05,
        random_state: int = 0,
        n_jobs: Union[None, int] = None,
        calc_knn: bool = True
) -> Tuple[float, float, float]:
    """Calculates the kBET metric of the data."""

    _logger.info('Calculating kbet...')
    assert batch_col in adata.obs
    
    # Ensure batch column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        adata.obs[batch_col] = adata.obs[batch_col].astype('category')
    
    # Get batch distribution and numerical codes
    batch_dist = adata.obs[batch_col].value_counts(normalize=True, sort=False)
    ideal_dist = batch_dist.values
    attr_values = adata.obs[batch_col].cat.codes.values
    n_batches = len(ideal_dist)

    # Get KNN indices
    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

    # Partition into chunks for parallel processing
    nsample = adata.shape[0]
    n_jobs = min(_eff_n_jobs(n_jobs), nsample)
    starts = np.zeros(n_jobs + 1, dtype=int)
    quotient = nsample // n_jobs
    remainder = nsample % n_jobs
    
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

    # Parallel computation
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", n_jobs=n_jobs):
        kBET_arr = np.concatenate(
            Parallel()(
                delayed(_calculate_kbet_for_one_chunk)(
                    knn_indices[starts[i]: starts[i + 1], :], 
                    attr_values, 
                    ideal_dist, 
                    n_neighbors
                )
                for i in range(n_jobs)
            )
        )

    # Compute final metrics
    stat_mean = kBET_arr[:, 0].mean()
    pvalue_mean = kBET_arr[:, 1].mean()
    accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample

    return (stat_mean, pvalue_mean, accept_rate)


# def calculate_kbet(
#         adata: ad.AnnData,
#         use_rep: str = "delta",
#         batch_col: str = "batch_indices",
#         n_neighbors: int = 25,
#         alpha: float = 0.05,
#         random_state: int = 0,
#         n_jobs: Union[None, int] = None,
#         calc_knn: bool = True
# ) -> Tuple[float, float, float]:
#     """Calculates the kBET metric of the data.

#     kBET measures if cells from different batches mix well in their local
#     neighborhood.

#     Args:
#         adata: annotated data matrix.
#         use_rep: the embedding to be used. Must exist in adata.obsm.
#         batch_col: a key in adata.obs to the batch column.
#         n_neighbors: # nearest neighbors.
#         alpha: acceptance rate threshold. A cell is accepted if its kBET
#             p-value is greater than or equal to alpha.
#         random_state: random seed. Used only if method is "hnsw".
#         n_jobs: # jobs to generate. If <= 0, this is set to the number of
#             physical cores.
#         calc_knn: whether to re-calculate the kNN graph or reuse the one stored
#             in adata.

#     Returns:
#         stat_mean: mean kBET chi-square statistic over all cells.
#         pvalue_mean: mean kBET p-value over all cells.
#         accept_rate: kBET Acceptance rate of the sample.
#     """

#     _logger.info('Calculating kbet...')
#     assert batch_col in adata.obs
#     if adata.obs[batch_col].dtype.name != "category":
#         _logger.warning(f'Making the column {batch_col} of adata.obs categorical.')
#         adata.obs[batch_col] = adata.obs[batch_col].astype('category')

#     ideal_dist = (
#         adata.obs[batch_col].value_counts(normalize=True, sort=False).values
#     )  # ideal no batch effect distribution
#     nsample = adata.shape[0]
#     nbatch = ideal_dist.size

#     attr_values = adata.obs[batch_col].values.copy()
#     attr_values.categories = range(nbatch)
#     knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

#     # partition into chunks
#     n_jobs = min(_eff_n_jobs(n_jobs), nsample)
#     starts = np.zeros(n_jobs + 1, dtype=int)
#     quotient = nsample // n_jobs
#     remainder = nsample % n_jobs
#     for i in range(n_jobs):
#         starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

#     from joblib import Parallel, delayed, parallel_backend
#     with parallel_backend("loky", n_jobs=n_jobs):
#         kBET_arr = np.concatenate(
#             Parallel()(
#                 delayed(_calculate_kbet_for_one_chunk)(
#                     knn_indices[starts[i]: starts[i + 1], :], attr_values, ideal_dist, n_neighbors
#                 )
#                 for i in range(n_jobs)
#             )
#         )

#     res = kBET_arr.mean(axis=0)
#     stat_mean = res[0]
#     pvalue_mean = res[1]
#     accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample

#     return (stat_mean, pvalue_mean, accept_rate)


def _entropy(hist_data):
    _, counts = np.unique(hist_data, return_counts=True)
    freqs = counts / counts.sum()
    return (-freqs * np.log(freqs + 1e-30)).sum()


def _entropy_batch_mixing_for_one_pool(batches, knn_indices, nsample, n_samples_per_pool):
    indices = np.random.choice(
        np.arange(nsample), size=n_samples_per_pool)
    return np.mean(
        [
            _entropy(batches[knn_indices[indices[i]]])
            for i in range(n_samples_per_pool)
        ]
    )


def calculate_entropy_batch_mixing(
        adata: ad.AnnData,
        use_rep: str = "delta",
        batch_col: str = "batch_indices",
        n_neighbors: int = 50,
        n_pools: int = 50,
        n_samples_per_pool: int = 100,
        random_state: int = 0,
        n_jobs: Union[None, int] = None,
        calc_knn: bool = True
) -> float:
    """Calculates the entropy of batch mixing of the data.

    kBET measures if cells from different batches mix well in their local
    neighborhood.

    Args:
        adata: annotated data matrix.
        use_rep: the embedding to be used. Must exist in adata.obsm.
        batch_col: a key in adata.obs to the batch column.
        n_neighbors: # nearest neighbors.
        n_pools: #pools of cells to calculate entropy of batch mixing.
        n_samples_per_pool: #cells per pool to calculate within-pool entropy.
        random_state: random seed. Used only if method is "hnsw".
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        calc_knn: whether to re-calculate the kNN graph or reuse the one stored
            in adata.

    Returns:
        score: the mean entropy of batch mixing, averaged from n_pools samples.
    """

    _logger.info('Calculating batch mixing entropy...')
    nsample = adata.n_obs

    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
        score = np.mean(
            Parallel()(
                delayed(_entropy_batch_mixing_for_one_pool)(
                    adata.obs[batch_col], knn_indices, nsample, n_samples_per_pool
                )
                for _ in range(n_pools)
            )
        )
    return score


def clustering(
        adata: ad.AnnData,
        resolutions: Sequence[float],
        clustering_method: str = "leiden",
        cell_type_col: str = "cell_type",
        batch_col: str = "batch_indices"
) -> Tuple[str, float, float]:
    """Clusters the data and calculate agreement with cell type and batch
    variable.

    This method cluster the neighborhood graph (requires having run sc.pp.
    neighbors first) with "clustering_method" algorithm multiple times with the
    given resolutions, and return the best result in terms of ARI with cell
    type.
    Other metrics such as NMI with cell type, ARi with batch are logged but not
    returned. (TODO: also return these metrics)

    Args:
        adata: the dataset to be clustered. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        resolutions: a list of leiden/louvain resolution parameters. Will
            cluster with each resolution in the list and return the best result
            (in terms of ARI with cell type).
        clustering_method: Either "leiden" or "louvain".
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.

    Returns:
        best_cluster_key: a key in adata.obs to the best (in terms of ARI with
            cell type) cluster assignment column.
        best_ari: the best ARI with cell type.
        best_nmi: the best NMI with cell type.
    """

    assert len(resolutions) > 0, f'Must specify at least one resolution.'

    if clustering_method == 'leiden':
        clustering_func = sc.tl.leiden
    elif clustering_method == 'louvain':
        clustering_func = sc.tl.louvain
    else:
        raise ValueError("Please specify louvain or leiden for the clustering method argument.")
    #_logger.info(f'Performing {clustering_method} clustering')
    assert cell_type_col in adata.obs, f"{cell_type_col} not in adata.obs"
    best_res, best_ari, best_nmi = None, -inf, -inf
    for res in resolutions:
        col = f'{clustering_method}_{res}'
        clustering_func(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs[cell_type_col], adata.obs[col])
        nmi = normalized_mutual_info_score(adata.obs[cell_type_col], adata.obs[col])
        n_unique = adata.obs[col].nunique()
        if ari > best_ari:
            best_res = res
            best_ari = ari
        if nmi > best_nmi:
            best_nmi = nmi
        if batch_col in adata.obs and adata.obs[batch_col].nunique() > 1:
            ari_batch = adjusted_rand_score(adata.obs[batch_col], adata.obs[col])
            #print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\tbARI: {ari_batch:7.4f}\t# labels: {n_unique}')
        else:
            #print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\t# labels: {n_unique}')
            a=None

    return f'{clustering_method}_{best_res}', best_ari, best_nmi


def draw_embeddings(adata: ad.AnnData,
                    color_by: Union[str, Sequence[str], None] = None,
                    min_dist: float = 0.3,
                    spread: float = 1,
                    ckpt_dir: str = '.',
                    fname: str = "umap.pdf",
                    return_fig: bool = False,
                    dpi: int = 300,
                    umap_kwargs: dict = dict()
                    ) -> Union[None, Figure]:
    """Embeds, plots and optionally saves the neighborhood graph with UMAP.

    Requires having run sc.pp.neighbors first.

    Args:
        adata: the dataset to draw. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        color_by: a str or a list of adata.obs keys to color the points in the
            scatterplot by. E.g. if both cell_type_col and batch_col is in
            color_by, then we would have two plots colored by cell type and
            batch variables, respectively.
        min_dist: The effective minimum distance between embedded points.
            Smaller values will result in a more clustered/clumped embedding
            where nearby points on the manifold are drawn closer together,
            while larger values will result on a more even dispersal of points.
        spread: The effective scale of embedded points. In combination with
            `min_dist` this determines how clustered/clumped the embedded
            points are.
        ckpt_dir: where to save the plot. If None, do not save the plot.
        fname: file name of the saved plot. Only used if ckpt_dir is not None.
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot.
        dpi: the dpi of the saved plot. Only used if ckpt_dir is not None.
        umap_kwargs: other kwargs to pass to sc.pl.umap.

    Returns:
        If return_fig is True, return the figure containing the plot.
    """

    #_logger.info(f'Plotting UMAP embeddings...')
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    fig = sc.pl.umap(adata, color=color_by, show=False, return_fig=True, **umap_kwargs)
    if ckpt_dir is not None:
        assert os.path.exists(ckpt_dir), f'ckpt_dir {ckpt_dir} does not exist.'
        fig.savefig(
            os.path.join(ckpt_dir, fname),
            dpi=dpi, bbox_inches='tight'
        )
    if return_fig:
        return fig
    fig.clf()
    plt.close(fig)


def set_figure_params(
        matplotlib_backend: str = 'agg',
        dpi: int = 120,
        frameon: bool = True,
        vector_friendly: bool = True,
        fontsize: int = 10,
        figsize: Sequence[int] = (10, 10)
):
    """Set figure parameters.
    Args
        backend: the backend to switch to.  This can either be one of th
            standard backend names, which are case-insensitive:
            - interactive backends:
                GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
                Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
                TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
            - non-interactive backends:
                agg, cairo, pdf, pgf, ps, svg, template
            or a string of the form: ``module://my.module.name``.
        dpi: resolution of rendered figures – this influences the size of
            figures in notebooks.
        frameon: add frames and axes labels to scatter plots.
        vector_friendly: plot scatter plots using `png` backend even when
            exporting as `pdf` or `svg`.
        fontsize: the fontsize for several `rcParams` entries.
        figsize: plt.rcParams['figure.figsize'].
    """
    matplotlib.use(matplotlib_backend)
    sc.set_figure_params(dpi=dpi, figsize=figsize, fontsize=fontsize, frameon=frameon, vector_friendly=vector_friendly)

def calculate_topic_coherence(topic_gene_matrix, top_k=10):
    """
    计算主题一致性
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        top_k: 每个主题考虑的顶部基因数量
        
    Returns:
        np.array: 每个主题的一致性分数
    """
    n_topics, n_genes = topic_gene_matrix.shape
    coherence_scores = []
    
    for topic_idx in range(n_topics):
        topic_weights = topic_gene_matrix[topic_idx]
        top_gene_indices = np.argsort(np.abs(topic_weights))[-top_k:]
        
        if len(top_gene_indices) < 2:
            coherence_scores.append(0.0)
            continue
            
        coherence_sum = 0
        pair_count = 0
        
        for i in range(len(top_gene_indices)):
            for j in range(i+1, len(top_gene_indices)):
                gene_i = top_gene_indices[i]
                gene_j = top_gene_indices[j]
                
                weights_i = topic_gene_matrix[:, gene_i]
                weights_j = topic_gene_matrix[:, gene_j]
                
                if np.std(weights_i) > 1e-8 and np.std(weights_j) > 1e-8:
                    correlation = np.corrcoef(weights_i, weights_j)[0, 1]
                    if not np.isnan(correlation):
                        coherence_sum += correlation
                        pair_count += 1
        
        coherence = coherence_sum / pair_count if pair_count > 0 else 0.0
        coherence_scores.append(coherence)
    
    return np.array(coherence_scores)

def calculate_topic_sparsity(topic_gene_matrix, threshold=None):
    """
    计算主题稀疏性
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        threshold: 判定显著基因的阈值,如果为None则自动计算
        
    Returns:
        np.array: 每个主题的稀疏性分数
    """
    # 如果没有提供阈值,使用每个主题的均值作为阈值
    if threshold is None:
        thresholds = np.mean(np.abs(topic_gene_matrix), axis=1, keepdims=True)
    else:
        thresholds = threshold * np.ones_like(topic_gene_matrix)
    
    # 计算每个主题中显著基因的比例
    significant_counts = np.sum(np.abs(topic_gene_matrix) > thresholds, axis=1)
    total_genes = topic_gene_matrix.shape[1]
    sparsity_scores = 1 - (significant_counts / total_genes)  # 转换为稀疏性分数
    
    return sparsity_scores

def calculate_topic_specificity(topic_gene_matrix):
    """
    计算主题特异性，基于Gini系数
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        
    Returns:
        np.array: 每个主题的特异性分数
    """
    n_topics, n_genes = topic_gene_matrix.shape
    specificity_scores = []
    
    for topic_idx in range(n_topics):
        topic_weights = np.abs(topic_gene_matrix[topic_idx])
        sorted_weights = np.sort(topic_weights)
        n = len(sorted_weights)
        cumulative_weights = np.cumsum(sorted_weights)
        
        if cumulative_weights[-1] > 1e-10:
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_weights))) / (n * cumulative_weights[-1]) - (n + 1) / n
        else:
            gini = 0.0
            
        specificity_scores.append(gini)
    
    return np.array(specificity_scores)

def calculate_topic_discreteness(cell_topic_matrix):
    """
    计算主题离散性
    
    Args:
        cell_topic_matrix: 细胞-主题矩阵
        
    Returns:
        tuple: (离散性总分, 每个主题的变异系数)
    """
    topic_means = np.mean(cell_topic_matrix, axis=0)
    topic_stds = np.std(cell_topic_matrix, axis=0)
    
    cv = np.divide(topic_stds, topic_means, out=np.zeros_like(topic_stds), where=topic_means!=0)
    discreteness = np.mean(cv)
    
    return discreteness, cv

def calculate_topic_entropy(topic_gene_matrix):
    """
    计算主题熵
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        
    Returns:
        np.array: 每个主题的熵值
    """
    from scipy.stats import entropy
    entropies = []
    
    for topic_idx in range(topic_gene_matrix.shape[0]):
        topic_weights = np.abs(topic_gene_matrix[topic_idx])
        total_weight = np.sum(topic_weights)
        if total_weight > 1e-10:
            topic_prob = topic_weights / total_weight
            topic_entropy = entropy(topic_prob + 1e-10)
        else:
            topic_entropy = 0.0
        entropies.append(topic_entropy)
    
    return np.array(entropies)

def calculate_topic_distance(topic_gene_matrix, method='cosine'):
    """
    计算主题之间的距离
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        method: 距离计算方法，可选'cosine', 'euclidean', 'jaccard'
        
    Returns:
        np.array: 主题间距离矩阵
    """
    if method == 'cosine':
        distances = pairwise_distances(topic_gene_matrix, metric='cosine')
    elif method == 'euclidean':
        distances = pairwise_distances(topic_gene_matrix, metric='euclidean')
    elif method == 'jaccard':
        threshold = np.percentile(np.abs(topic_gene_matrix), 50)
        binary_matrix = (np.abs(topic_gene_matrix) > threshold).astype(int)
        distances = pairwise_distances(binary_matrix, metric='jaccard')
    else:
        raise ValueError(f"Unsupported distance method: {method}")
    
    return distances

def calculate_topic_diversity(topic_gene_matrix):
    """
    计算主题多样性
    
    Args:
        topic_gene_matrix: 主题-基因矩阵
        
    Returns:
        float: 主题多样性分数
    """
    distances = calculate_topic_distance(topic_gene_matrix, method='cosine')
    upper_tri_indices = np.triu_indices(distances.shape[0], k=1)
    if len(upper_tri_indices[0]) > 0:
        diversity = np.mean(distances[upper_tri_indices])
    else:
        diversity = 0.0
    return diversity

def calculate_cross_modal_alignment(topic_gene_mod1, topic_gene_mod2):
    """
    计算跨模态主题对齐性
    
    Args:
        topic_gene_mod1: 第一个模态的主题-基因矩阵
        topic_gene_mod2: 第二个模态的主题-基因矩阵
        
    Returns:
        np.array: 每个主题对的对齐分数
    """
    alignment_scores = []
    n_topics = min(topic_gene_mod1.shape[0], topic_gene_mod2.shape[0])
    
    for topic_idx in range(n_topics):
        weights_mod1 = topic_gene_mod1[topic_idx]
        weights_mod2 = topic_gene_mod2[topic_idx]
        
        weights_mod1_norm = (weights_mod1 - np.mean(weights_mod1)) / (np.std(weights_mod1) + 1e-10)
        weights_mod2_norm = (weights_mod2 - np.mean(weights_mod2)) / (np.std(weights_mod2) + 1e-10)
        
        moments_mod1 = [
            np.mean(weights_mod1_norm),
            np.std(weights_mod1_norm),
            np.mean(weights_mod1_norm**3),
            np.mean(weights_mod1_norm**4)
        ]
        
        moments_mod2 = [
            np.mean(weights_mod2_norm),
            np.std(weights_mod2_norm),
            np.mean(weights_mod2_norm**3),
            np.mean(weights_mod2_norm**4)
        ]
        
        moments_mod1 = np.array(moments_mod1)
        moments_mod2 = np.array(moments_mod2)
        
        norm1 = np.linalg.norm(moments_mod1)
        norm2 = np.linalg.norm(moments_mod2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            similarity = np.dot(moments_mod1, moments_mod2) / (norm1 * norm2)
        else:
            similarity = 0.0
            
        alignment_scores.append(similarity)
    
    return np.array(alignment_scores)
