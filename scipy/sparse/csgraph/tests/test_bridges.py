import sys

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import scipy.sparse
from scipy.sparse import (
    bsr_array, coo_array, csc_array, csr_array, csr_matrix, dok_array,
    lil_array,
)
import scipy.sparse.csgraph as csgraph
from scipy.sparse.csgraph import bridges, connected_components


# --------------------------------------------------------------------------
# helpers and a brute-force reference implementation
# --------------------------------------------------------------------------

def _as_csgraph_csr(graph):
    """Convert any accepted input to CSR using the csgraph edge semantics:

    dense 0 / inf / nan entries are non-edges; masked entries are
    non-edges; every stored sparse entry (including explicit zeros) is
    an edge.
    """
    if np.ma.isMaskedArray(graph):
        return csgraph.csgraph_from_masked(graph)
    if scipy.sparse.issparse(graph):
        return graph.tocsr(copy=False)
    return csgraph.csgraph_from_dense(np.asarray(graph), null_value=0)


def _undirected_edges(csr):
    coo = csr.tocoo()
    pairs = {
        (min(int(r), int(c)), max(int(r), int(c)))
        for r, c in zip(coo.row, coo.col)
        if r != c
    }
    return sorted(pairs)


def _drop_edge(csr, a, b):
    coo = csr.tocoo()
    keep = ~(((coo.row == a) & (coo.col == b))
             | ((coo.row == b) & (coo.col == a)))
    return csr_array((coo.data[keep], (coo.row[keep], coo.col[keep])),
                     shape=csr.shape)


def reference_bridges(graph):
    """Brute-force O(E * (V + E)) bridge computation via component counts.
    Only used on tiny graphs in the tests."""
    csr = _as_csgraph_csr(graph)
    n_components, _ = connected_components(csr, directed=False)
    out = []
    for a, b in _undirected_edges(csr):
        n2, _ = connected_components(_drop_edge(csr, a, b), directed=False)
        if n2 > n_components:
            out.append([a, b])
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def assert_bridges(result, expected):
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert np.issubdtype(result.dtype, np.integer)
    if expected is None or len(expected) == 0:
        assert result.shape == (0, 2)
        return
    expected = np.asarray(expected)
    assert_array_equal(result, expected)
    # the smaller endpoint is listed first
    assert np.all(result[:, 0] < result[:, 1])
    # rows are unique and lexicographically sorted
    assert_array_equal(result, np.unique(result, axis=0))


# --------------------------------------------------------------------------
# basic structures
# --------------------------------------------------------------------------

def test_path_asymmetric():
    # only the upper direction is stored: the graph is still a path
    path = np.array([[0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]])
    assert_bridges(bridges(path), [[0, 1], [1, 2], [2, 3]])


def test_path_symmetric():
    path = np.array([[0, 1, 0, 0],
                     [1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0]])
    assert_bridges(bridges(path), [[0, 1], [1, 2], [2, 3]])


def test_star():
    n = 5
    edges = [(i, 0) for i in range(1, n)]
    rows = [r for r, _ in edges]
    cols = [c for _, c in edges]
    g = csr_array((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    # every spoke is a bridge; output must be lexicographically ordered
    assert_bridges(bridges(g), [[0, 1], [0, 2], [0, 3], [0, 4]])


def test_cycle():
    cycle = np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0]])
    assert_bridges(bridges(cycle), [])

    triangle = np.array([[0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 0]])
    assert_bridges(bridges(triangle), [])


def test_cycle_with_path_attached():
    # triangle (0,1,2) with a path 2-3-4 attached at node 2
    pan = np.array([[0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                    [1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]])
    assert_bridges(bridges(pan), [[2, 3], [3, 4]])


def test_two_cycles_joined_by_path():
    # triangles on {0,1,2} and {5,6,7}, joined by path 2-3-4-5
    g = np.zeros((8, 8), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 0),
                 (2, 3), (3, 4), (4, 5),
                 (5, 6), (6, 7), (7, 5)]:
        g[a, b] = g[b, a] = 1
    assert_bridges(bridges(g), [[2, 3], [3, 4], [4, 5]])


def test_disconnected_components():
    # 4-cycle on 0..3, path 4-5-6, isolated node 7
    g = np.zeros((8, 8), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)]:
        g[a, b] = 1  # one direction only
    assert_bridges(bridges(g), [[4, 5], [5, 6]])


# --------------------------------------------------------------------------
# bidirectional edges count once
# --------------------------------------------------------------------------

def test_bidirectional_edge_is_one_bridge():
    g = np.array([[0, 1],
                  [1, 0]])
    # the two stored directions represent one undirected edge, and it is
    # a bridge; it must appear exactly once
    assert_bridges(bridges(g), [[0, 1]])


def test_bidirectional_path_counted_once():
    g = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    result = bridges(g)
    assert_bridges(result, [[0, 1], [1, 2]])


def test_bidirectional_triangle_has_no_bridge():
    g = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    result = bridges(csr_array(g))
    assert result.shape == (0, 2)


def test_mixed_directions():
    # 0 <-> 1 in both directions, 1 -> 2 in one direction only:
    # both undirected edges are bridges
    g = csr_array(np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 0, 0]]))
    assert_bridges(bridges(g), [[0, 1], [1, 2]])


# --------------------------------------------------------------------------
# self loops, empty graphs, isolated nodes
# --------------------------------------------------------------------------

def test_self_loops_are_not_bridges():
    g = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 9, 1],
                  [0, 0, 1, 1]])
    assert_bridges(bridges(g), [[0, 1], [1, 2], [2, 3]])

    only_loop = np.array([[1, 0],
                          [0, 1]])
    assert_bridges(bridges(only_loop), [])


@pytest.mark.parametrize("graph", [
    np.zeros((0, 0)),
    csr_array((0, 0)),
    np.zeros((5, 5)),
    csr_array((5, 5)),
    np.ma.masked_all((4, 4)),
])
def test_no_edges(graph):
    result = bridges(graph)
    assert result.dtype.kind in "iu"
    assert result.shape == (0, 2)


def test_empty_graph_dtype():
    result = bridges(np.zeros((0, 0)))
    assert result.shape == (0, 2)


# --------------------------------------------------------------------------
# storage formats, dtypes and null values
# --------------------------------------------------------------------------

def _example_graph():
    # triangle (0,1,2), tail 2-3, self loop on 1, isolated node 4
    return np.array([[0, 1, 1, 0, 0],
                     [1, 7, 1, 0, 0],
                     [1, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])


@pytest.mark.parametrize("converter", [
    lambda g: g,
    lambda g: np.asarray(g, dtype=np.float64),
    lambda g: np.asarray(g, dtype=np.int64),
    lambda g: np.ma.masked_values(g, 0),
    lambda g: csr_array(g),
    lambda g: csc_array(g),
    lambda g: coo_array(g),
    lambda g: lil_array(g),
    lambda g: dok_array(g),
    lambda g: bsr_array(g),
    lambda g: csr_matrix(g),
])
def test_storage_formats_and_dtypes(converter):
    g = converter(_example_graph())
    assert_bridges(bridges(g), [[2, 3]])


def test_bool_dense_and_sparse():
    g = np.array([[False, True, False],
                  [True, False, True],
                  [False, True, False]])
    assert_bridges(bridges(g), [[0, 1], [1, 2]])
    assert_bridges(bridges(csr_array(g)), [[0, 1], [1, 2]])


def test_sparse_explicit_zero_is_edge():
    # the (0, 1) edge has weight zero, but it is stored and thus exists
    g = csr_array(([0.0, 1.0], ([0, 2], [1, 0])), shape=(3, 3))
    assert g.nnz == 2
    assert_bridges(bridges(g), [[0, 1], [0, 2]])

    # explicit zero completing a cycle prevents any bridge
    cycle = csr_array(([0.0, 1.0, 1.0], ([0, 1, 2], [1, 2, 0])),
                      shape=(3, 3))
    assert bridges(cycle).shape == (0, 2)


def test_dense_zero_is_not_edge():
    # in dense representation plain zeros are non-edges, even though a
    # stored sparse zero would be an edge
    dense = np.zeros((2, 2))
    assert bridges(dense).shape == (0, 2)

    sparse_stored_zero = csr_array(([0.0], ([0], [1])), shape=(2, 2))
    assert_bridges(bridges(sparse_stored_zero), [[0, 1]])


def test_dense_inf_and_nan_are_non_edges():
    g = np.array([[np.inf, 1.0],
                  [0.0, np.nan]])
    assert_bridges(bridges(g), [[0, 1]])


# --------------------------------------------------------------------------
# invalid input and non-mutation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("graph", [
    np.zeros((2, 3)),
    csr_array((2, 3)),
    csc_array((2, 3)),
    np.zeros(4),
])
def test_invalid_shapes_raise(graph):
    with pytest.raises(ValueError):
        bridges(graph)


def test_input_not_modified_csr():
    g = csr_array(np.array([[0, 1, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 0, 1]], dtype=np.float64))
    data, indices, indptr = g.data.copy(), g.indices.copy(), g.indptr.copy()
    bridges(g)
    assert_array_equal(g.data, data)
    assert_array_equal(g.indices, indices)
    assert_array_equal(g.indptr, indptr)
    assert g.nnz == len(data)


def test_input_not_modified_csc():
    g = csc_array(_example_graph())
    data, indices, indptr = g.data.copy(), g.indices.copy(), g.indptr.copy()
    bridges(g)
    assert_array_equal(g.data, data)
    assert_array_equal(g.indices, indices)
    assert_array_equal(g.indptr, indptr)


def test_input_not_modified_dense():
    g = _example_graph()
    snapshot = g.copy()
    bridges(g)
    assert_array_equal(g, snapshot)


def test_explicit_zeros_not_eliminated():
    g = csr_array(([0.0, 0.0, 1.0], ([0, 1, 2], [1, 2, 0])),
                  shape=(3, 3))
    nnz_before = g.nnz
    bridges(g)
    assert g.nnz == nnz_before


def test_repeated_zero_slots_do_not_vanish():
    # hand-built, non-canonical CSR with 256 explicit-zero slots for the
    # same off-diagonal pair; pattern union must not overflow a narrow
    # accumulator and lose the edge
    n_slots = 256
    g = csr_array((2, 2))
    g.data = np.zeros(n_slots, dtype=np.float64)
    g.indices = np.full(n_slots, 1, dtype=np.int32)
    g.indptr = np.array([0, n_slots, n_slots], dtype=np.int32)
    assert_bridges(bridges(g), [[0, 1]])


# --------------------------------------------------------------------------
# iterative traversal must not depend on the recursion limit
# --------------------------------------------------------------------------

def test_long_path_does_not_hit_recursion_limit():
    n = 5000
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        g = csr_array(
            (np.ones(n - 1), (np.arange(n - 1), np.arange(1, n))),
            shape=(n, n))
        result = bridges(g)
    finally:
        sys.setrecursionlimit(old_limit)
    expected = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    assert_array_equal(result, expected)


# --------------------------------------------------------------------------
# randomized cross-validation against the brute-force reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_random_dense_crosscheck(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 9)
    g = rng.integers(0, 2, size=(n, n))
    np.fill_diagonal(g, rng.integers(0, 2, size=n))  # self loops
    assert_bridges(bridges(g), reference_bridges(g))


@pytest.mark.parametrize("seed", range(20))
def test_random_sparse_crosscheck(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 9)
    # random stored entries, including diagonal ones and explicit zero
    # weights, and asymmetric patterns
    positions = [tuple(p) for p in np.argwhere(
        rng.random((n, n)) < 0.4)]
    if not positions:
        positions = [(0, 0)]
    rows = np.array([p[0] for p in positions])
    cols = np.array([p[1] for p in positions])
    data = rng.integers(0, 3, size=len(positions)).astype(np.float64)
    g = csr_array((data, (rows, cols)), shape=(n, n))
    assert_bridges(bridges(g), reference_bridges(g))


@pytest.mark.parametrize("seed", range(10))
def test_random_sparse_format_crosscheck(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 8)
    dense = rng.integers(0, 2, size=(n, n))
    for make in (csr_array, csc_array, coo_array, lil_array,
                 lambda x: np.ma.masked_values(x, 0)):
        g = make(dense)
        assert_bridges(bridges(g), reference_bridges(g))
