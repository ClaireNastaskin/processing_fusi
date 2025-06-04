"""Detect motion artifacts in ultrasound ensembles.

Uses the correlation between I/Q data across time, or across images.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from einops import rearrange
from jaxtyping import Array, Int, Num, PRNGKeyArray, Real, jaxtyped
import h5py

@jaxtyped(typechecker=typechecker)
def pairwise_corrcoef_agg(
    vectors: Num[Array, " n_vectors *each_vector_dims"],
) -> float:
    """Aggregates pairwise-correlation-coefficients between all vectors.

    Note: makes an opinionated choice to manage complex correlation coefficients,
        assuming that the real-part is the important component. This should be
        reasonable for data that is expected to be in-phase.

    Args:
        vectors: Array of shape (n_vectors, ...) containing the vectors to correlate
            If each vector is multidimensional, it will be flattened to a 1D array.
    Returns:
        Aggregation across all pairwise-correlation-coefficients
    """
    corrcoefs, _ = pairwise_corrcoef(vectors=vectors)
    return float(jnp.real(jnp.mean(corrcoefs)))


@jaxtyped(typechecker=typechecker)
def pairwise_corrcoef(
    vectors: Num[Array, " n_vectors *each_vector_dims"],
    n_pairs: Optional[int] = None,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[Num[Array, " n_pairs"], Int[Array, "n_pairs 2"]]:
    """Compute pairwise-correlation-coefficients between vectors efficiently using JAX.

    Args:
        vectors: Array of shape (n_vectors, ...) containing the vectors to correlate
            If each vector is multidimensional, it will be flattened to a 1D array.
        n_pairs: Optional number of random pairs to compute. If None, computes all pairs.
        key: JAX PRNG key for random sampling. If None, uses PRNGKey(0).

    Returns:
        tuple of (correlations, pairs):
            correlations: Array of correlation coefficients
            pairs: Array of shape (n_pairs, 2) containing the indices of correlated pairs
    """
    # If n_pairs is None or equal to max_pairs, compute all pairs
    if n_pairs is None:
        return pairwise_corrcoef_all_jit(vectors)

    tril_indices_tup = jnp.tril_indices(vectors.shape[0], k=-1)
    pairs = jnp.column_stack(tup=tril_indices_tup)
    max_pairs = pairs.shape[0]
    if n_pairs >= max_pairs:
        return pairwise_corrcoef_all_jit(vectors)

    # Randomly select index-pairs outside of JIT
    # because JIT cannot handle dynamically-sized arrays
    assert n_pairs is not None
    if key is None:
        key = jax.random.PRNGKey(0)
    indices = jax.random.permutation(key, max_pairs)[:n_pairs]
    pairs = pairs[indices]

    return pairwise_corrcoef_jit(vectors, pairs)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def pairwise_corrcoef_all_jit(
    vectors: Num[Array, " n_vectors *each_vector_dims"],
) -> tuple[Num[Array, " n_pairs"], Int[Array, "n_pairs 2"]]:
    """Compute pairwise-correlation-coefficients between all vectors with JIT-compiled JAX.
    Args:
        vectors: Array of shape (n_vectors, ...) containing the vectors to correlate
            If each vector is multidimensional, it will be flattened to a 1D array.
    Returns:
        tuple of (correlations, pairs):
            correlations: Array of correlation coefficients
            pairs: Array of shape (n_pairs, 2) containing the indices of correlated pairs
    """
    n_vectors = vectors.shape[0]
    vectors = rearrange(tensor=vectors, pattern="n_vectors ... -> n_vectors (...)")
    correlation_matrix = corrcoef(vectors)
    tril_indices_tup = jnp.tril_indices(n_vectors, k=-1)
    return correlation_matrix[tril_indices_tup], jnp.column_stack(tril_indices_tup)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def corrcoef(vectors: Num[Array, "n_observations n_features"]) -> Num[Array, " n_observations n_observations"]:
    """Fast, simplified implementation of jnp.corrcoef.

    Args:
        vectors: Input data with shape (n_observations, n_features)
            e.g. each observation is a B-Mode within an ensemble
            and each feature is a voxel in the B-Mode

    Returns:
        Correlation-coefficient matrix
    """
    vectors_centered = vectors - jnp.mean(vectors, axis=1)[:, None]
    # Use ddof=1 for unbiased estimator, to match numpy's default
    vectors_normalized = vectors_centered / jnp.std(vectors, axis=1, ddof=1)[:, None]
    corr = jnp.matmul(vectors_normalized, vectors_normalized.conj().T) / (vectors.shape[1] - 1)
    return _clip_corrcoef(corrcoefs=corr)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def pairwise_corrcoef_jit(
    vectors: Num[Array, " n_vectors *each_vector_dims"],
    pairs: Int[Array, " n_pairs 2"],
) -> tuple[Num[Array, " n_pairs"], Int[Array, "n_pairs 2"]]:
    """Compute subset of pairwise-correlation-coefficients between vectors with JIT-compiled JAX.

    Args:
        vectors: Array of shape (n_vectors, ...) containing the vectors to correlate
            If each vector is multidimensional, it will be flattened to a 1D array.
        pairs: Array of shape (n_pairs, 2) containing the indices of pairs
            to compute correlations for.

    Returns:
        tuple of (correlations, pairs):
            correlations: Array of correlation coefficients
            pairs: Array of shape (n_pairs, 2) containing the indices of correlated pairs
    """
    # Flatten all dimensions after the first into a single dimension
    vectors = rearrange(vectors, "n_vectors ... -> n_vectors (...)")

    # Normalize vectors for faster computation
    # Use ddof=1 for unbiased estimator, to match later normalization
    vectors_normalized = (vectors - jnp.mean(vectors, axis=1)[:, None]) / jnp.std(vectors, axis=1, ddof=1)[:, None]

    # Compute correlations efficiently using dot product of normalized vectors
    correlations = (
        jnp.sum(vectors_normalized[pairs[:, 0]] * vectors_normalized[pairs[:, 1]].conj(), axis=1) / (vectors.shape[1])
    )

    correlations = _clip_corrcoef(corrcoefs=correlations)
    return correlations, pairs


def _clip_corrcoef(corrcoefs: Num[Array, "*sizes"], tol: float = 0.2) -> Num[Array, "*sizes"]:
    """Clip correlation coefficients to [-1, 1] to avoid numerical errors.

    Args:
        corrcoefs: Array of correlation coefficients
        tol: Tolerance for values slightly outside of [-1, 1]
    """
    if jnp.iscomplexobj(corrcoefs):
        return _clip_corrcoef_complex(corrcoefs=corrcoefs, tol=tol)
    else:
        return _clip_corrcoef_real(corrcoefs=corrcoefs, tol=tol)


@jaxtyped(typechecker=typechecker)
def _clip_corrcoef_real(corrcoefs: Real[Array, "*sizes"], tol: float = 0.01) -> Real[Array, "*sizes"]:
    """Clip correlation coefficients to [-1, 1] to avoid numerical errors.

    Args:
        corrcoefs: Array of correlation coefficients
        tol: Tolerance for values slightly outside of [-1, 1]

    Returns:
        corrcoefs, clipped to [-1, 1]
    """
    corrcoefs = eqx.error_if(
        x=corrcoefs,
        pred=corrcoefs < -(1 + tol),
        msg="Correlation coefficient below -1",
    )
    corrcoefs = eqx.error_if(
        x=corrcoefs,
        pred=corrcoefs > 1 + tol,
        msg="Correlation coefficient above 1",
    )
    corrcoefs = jnp.clip(corrcoefs, min=-1, max=1)
    return corrcoefs


@jaxtyped(typechecker=typechecker)
def _clip_corrcoef_complex(corrcoefs: Num[Array, "*sizes"], tol: float = 0.01) -> Num[Array, "*sizes"]:
    """Clip correlation coefficients to [-1, 1] to avoid numerical errors.

    Args:
        corrcoefs: Array of correlation coefficients
        tol: Tolerance for values slightly outside of [-1, 1]
    """
    real_corrcoefs = jnp.real(corrcoefs)
    imag_corrcoefs = jnp.imag(corrcoefs)
    real_corrcoefs = _clip_corrcoef_real(corrcoefs=real_corrcoefs, tol=tol)
    imag_corrcoefs = _clip_corrcoef_real(corrcoefs=imag_corrcoefs, tol=tol)
    return real_corrcoefs + 1j * imag_corrcoefs


if __name__ == "__main__":
    import time

    import jax.numpy as jnp

    # # Example usage:
    # print("Example usage of pairwise_corrcoef")
    # key = jax.random.PRNGKey(0)
    # # Example 200-ensemble, 8960-channel, 60-timepoint data
    # vectors = jax.random.normal(key, (200, 8_960, 60), dtype=jnp.complex64)
    # # Example: Use all pairs
    # n_pairs = None
    # print(f"Vectors size and info: {vectors.shape} {vectors.dtype} {vectors.nbytes / 1e6} MB")

    # # First call will compile
    # start = time.time()
    # correlations, pairs = pairwise_corrcoef(vectors, n_pairs=n_pairs)
    # end = time.time()
    # print(f"First call took: {end - start} seconds")
    # print(f"Correlations shape: {correlations.shape}")

    # # Subsequent calls will be much faster
    # start = time.time()
    # correlations, pairs = pairwise_corrcoef(vectors, n_pairs=n_pairs)
    # end = time.time()
    # print(f"Second call took: {end - start} seconds")
    # print(f"Correlations shape: {correlations.shape}")

    # Example usage:
    print("Example usage of pairwise_corrcoef")
    key = jax.random.PRNGKey(0)


    # Example 200-ensemble, 8960-channel, 60-timepoint data
    # vectors = jax.random.normal(key, (200, 8_960, 60), dtype=jnp.complex64)
    h5FilePath = '/Users/clairenastaskin/data/2025-04-10/sub-Forest001_ses-20250410_task-movements_run-02/70mm_80mm/raw_data_not_encoded/schema/ensemble_2025-04-10T18-43-50-726461.h5'
    with h5py.File(h5FilePath, 'r') as f:
        vectors = f['data'][:]
    # Convert NumPy array to JAX array
    vectors = jnp.array(vectors)
    # Example: Use all pairs 
    n_pairs = None
    # print(f"Vectors size and info: {vectors.shape} {vectors.dtype} {vectors.nbytes / 1e6} MB")

    # First call will compile
    start = time.time()
    correlation = pairwise_corrcoef_agg(vectors)
    end = time.time()
    print(f"First call took: {end - start} seconds")
    print(f"Correlation value: {correlation}")

    # Subsequent calls will be much faster
    start = time.time()
    correlation = pairwise_corrcoef_agg(vectors)
    end = time.time()
    print(f"Second call took: {end - start} seconds")
    print(f"Correlation value: {correlation}")