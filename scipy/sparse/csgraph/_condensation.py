"""
Condensation of a directed compressed-sparse graph into its
strongly connected components.
"""

import numpy as np
from scipy.sparse import coo_array, csr_array

from scipy.sparse.csgraph._traversal import connected_components
from scipy.sparse.csgraph._validation import validate_graph

__all__ = ['condensation']


def condensation(csgraph):
    """
    condensation(csgraph)

    Condense a directed graph into the directed acyclic graph of its
    strongly connected components.

    Every strongly connected component (SCC) of the input graph becomes a
    single vertex of the returned component graph.  A directed edge is
    present from component *i* to component *j* if at least one original
    directed edge runs from a vertex in *i* to a vertex in *j*.  The value
    stored for each component-graph edge is the number of distinct
    original directed edges between the two components.  Edges internal to
    a component (including self-loops) do not appear in the result.

    The rules for which entries of the input are considered edges are the
    same as for :func:`connected_components` with ``directed=True`` and
    ``connection='strong'``: in particular, explicitly stored zeros of a
    sparse graph count as edges, whereas zeros of a dense array do not.

    Parameters
    ----------
    csgraph : array_like or sparse array or matrix
        The N x N matrix representing the directed compressed sparse
        graph.  The input is converted to CSR format for the calculation
        and is not modified.

    Returns
    -------
    labels : ndarray, shape (N,), dtype=int32
        Component number of each of the N original vertices.  Components
        are numbered in increasing order of the smallest vertex index
        they contain: the component containing vertex 0 has label 0, the
        component containing the smallest vertex not in component 0 has
        label 1, and so on.
    component_graph : csr_array, shape (n_components, n_components)
        The condensed directed acyclic graph.  ``component_graph[i, j]``
        is the number of distinct directed edges of the original graph
        that go from a vertex of component *i* to a vertex of component
        *j*.  There are no diagonal entries and, by construction, the
        component graph contains no directed cycles.  For an empty input
        graph, this is a ``(0, 0)`` CSR array.

    Raises
    ------
    ValueError
        If the input graph is not a square 2-D matrix.

    Notes
    -----
    The construction uses only sparse intermediates: neither the
    N x N vertex adjacency matrix nor an n_components x n_components
    component matrix is ever materialized in dense form.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> from scipy.sparse.csgraph import condensation

    Two disjoint two-cycles linked by a single directed edge:

    >>> graph = csr_array(np.array([[0, 1, 0, 0, 0],
    ...                             [1, 0, 1, 0, 0],
    ...                             [0, 0, 0, 1, 0],
    ...                             [0, 0, 1, 0, 0],
    ...                             [0, 0, 0, 0, 0]]))
    >>> labels, component_graph = condensation(graph)
    >>> labels
    array([0, 0, 1, 1, 2], dtype=int32)
    >>> component_graph.toarray()
    array([[0, 1, 0],
           [0, 0, 0],
           [0, 0, 0]])

    """
    # Validate exactly like connected_components(directed=True,
    # connection='strong'), so edge-existence semantics match (sparse
    # explicit zeros are kept; dense zeros are not).
    csgraph = validate_graph(csgraph, True, dtype=csgraph.dtype,
                             dense_output=False)

    n = csgraph.shape[0]
    if n == 0:
        return (np.empty(0, dtype=np.int32),
                csr_array((0, 0), dtype=np.int64))

    # Delegate SCC detection to connected_components so the edge-existence
    # rules (including sparse explicit zeros) match it by construction.
    # The graph is already a validated CSR, so this does not copy it.
    n_components, labels = connected_components(
        csgraph, directed=True, connection='strong'
    )

    # Number components in increasing order of their smallest member
    # vertex (Tarjan emits them in root post-order, which is unrelated to
    # vertex index order).
    vertex_index = np.arange(n)
    min_vertex = np.full(n_components, n, dtype=np.int64)
    np.minimum.at(min_vertex, labels, vertex_index)
    remap = np.empty(n_components, dtype=np.int32)
    remap[np.argsort(min_vertex)] = np.arange(n_components, dtype=np.int32)
    labels = remap[labels]

    # Expand CSR structure into (row, col) edge coordinates without
    # touching the input data: the edge pattern is fully described by
    # indptr/indices, so stored zeros are treated like any other edge.
    # The columns are copied: they are fed to a COO container whose
    # sum_duplicates() may reorder them in place, and the input must
    # never be modified.
    indptr = csgraph.indptr
    edge_rows = np.repeat(
        np.arange(n, dtype=csgraph.indices.dtype),
        np.diff(indptr)
    )
    edge_cols = csgraph.indices.copy()

    # Coalesce duplicate coordinates: a repeatedly stored directed
    # coordinate counts as a single edge.  This stays sparse (COO sort,
    # no dense intermediate).
    edges = coo_array(
        (np.ones(edge_rows.shape[0], dtype=np.int64),
         (edge_rows, edge_cols)),
        shape=(n, n)
    )
    edges.sum_duplicates()

    # Project the distinct edges onto component pairs, dropping edges
    # whose endpoints belong to the same component (self-loops included).
    src_components = labels[edges.row]
    dst_components = labels[edges.col]
    cross = src_components != dst_components

    # Count the distinct edges between each ordered pair of components.
    component_graph = coo_array(
        (np.ones(int(np.count_nonzero(cross)), dtype=np.int64),
         (src_components[cross], dst_components[cross])),
        shape=(n_components, n_components)
    )
    component_graph.sum_duplicates()

    return labels, component_graph.tocsr()
