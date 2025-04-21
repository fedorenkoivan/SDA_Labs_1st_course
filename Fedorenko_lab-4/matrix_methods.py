import numpy as np

def calculate_degrees(matrix, is_directed=True):
    """Calculate vertex degrees
    For directed graphs, returns in-degrees and out-degrees
    For undirected graphs, returns degrees
    """
    n = matrix.shape[0]
    
    if is_directed:
        in_degrees = np.sum(matrix, axis=0)
        out_degrees = np.sum(matrix, axis=1)
        return in_degrees, out_degrees
    else:
        degrees = np.sum(matrix, axis=1)
        return degrees

def generate_adjacency_matrix(n, variant_number, k_formula=1):
    """Generate directed adjacency matrix based on variant number and k_formula
    k_formula=1: k = 1.0 - n3 * 0.01 - n4 * 0.01 - 0.3
    k_formula=2: k = 1.0 - n3 * 0.005 - n4 * 0.005 - 0.27
    """
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    
    n3 = 2
    n4 = 9
    
    if k_formula == 1:
        k = 1.0 - n3 * 0.01 - n4 * 0.01 - 0.3
    else:
        k = 1.0 - n3 * 0.005 - n4 * 0.005 - 0.27
    
    print(f"Using k coefficient: {k}")
    
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 if T[i, j] * k >= 1.0 else 0
    
    return A

def get_undirected_matrix(directed_matrix):
    """Convert directed adjacency matrix to undirected"""
    n = directed_matrix.shape[0]
    undirected_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if directed_matrix[i, j] == 1 or directed_matrix[j, i] == 1:
                undirected_matrix[i, j] = 1
                undirected_matrix[j, i] = 1
    
    return undirected_matrix

def calculate_strong_connectivity_matrix(reachability):
    """Calculate strong connectivity matrix from reachability matrix"""
    n = reachability.shape[0]
    strong_connectivity = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if reachability[i, j] == 1 and reachability[j, i] == 1:
                strong_connectivity[i, j] = 1
    
    return strong_connectivity

def print_matrix(matrix, title):
    """Print the adjacency matrix in a readable format"""
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def calculate_reachability_matrix(matrix):
    """Calculate reachability matrix using transitive closure"""
    n = matrix.shape[0]
    
    # Initialize reachability with the adjacency matrix
    reachability = matrix.copy()
    
    # Add self-loops
    for i in range(n):
        reachability[i, i] = 1
    
    # Warshall's algorithm for transitive closure
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reachability[i, j] = reachability[i, j] or (reachability[i, k] and reachability[k, j])
    
    return reachability