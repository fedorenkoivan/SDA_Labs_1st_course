import numpy as np
from matrix_methods import calculate_degrees

def is_regular_graph(degrees):
    """Check if graph is regular (all vertices have the same degree)"""
    return np.all(degrees == degrees[0])

def find_special_vertices(matrix, is_directed=True):
    """Find hanging (leaf) and isolated vertices"""
    n = matrix.shape[0]
    
    if is_directed:
        in_degrees, out_degrees = calculate_degrees(matrix, is_directed=True)
        total_degrees = in_degrees + out_degrees
        
        isolated = [i+1 for i in range(n) if in_degrees[i] == 0 and out_degrees[i] == 0]
        hanging = [i+1 for i in range(n) if total_degrees[i] == 1]
    else:
        degrees = calculate_degrees(matrix, is_directed=False)
        
        isolated = [i+1 for i in range(n) if degrees[i] == 0]
        hanging = [i+1 for i in range(n) if degrees[i] == 1]
    
    return hanging, isolated

def find_paths_of_length(matrix, length):
    """Find all paths of specific length using matrix powers"""
    n = matrix.shape[0]
    
    power_matrix = np.linalg.matrix_power(matrix, length)
    
    paths = []
    for i in range(n):
        for j in range(n):
            if power_matrix[i, j] > 0:
                intermediate_paths = find_all_paths(matrix, i, j, length)
                paths.extend(intermediate_paths)
    
    return paths

def find_all_paths(matrix, start, end, length, current_path=None, current_length=0):
    """Recursively find all paths of a specific length from start to end"""
    n = matrix.shape[0]
    
    if current_path is None:
        current_path = [start]
    
    if current_length == length:
        if current_path[-1] == end:
            return [current_path]
        return []
    
    paths = []
    for next_vertex in range(n):
        if matrix[current_path[-1], next_vertex] == 1:
            if current_length == length - 1 and next_vertex == end:
                paths.append(current_path + [next_vertex])
            elif current_length < length - 1:
                new_paths = find_all_paths(
                    matrix, start, end, length, 
                    current_path + [next_vertex], 
                    current_length + 1
                )
                paths.extend(new_paths)
    
    return paths

def find_strongly_connected_components(strong_connectivity):
    """Find strongly connected components from strong connectivity matrix"""
    n = strong_connectivity.shape[0]
    visited = [False] * n
    components = []
    
    for vertex in range(n):
        if not visited[vertex]:
            component = []
            dfs_component(vertex, strong_connectivity, visited, component)
            components.append([v+1 for v in component])  # Convert to 1-indexed
    
    return components

def dfs_component(vertex, matrix, visited, component):
    """DFS to find connected components"""
    visited[vertex] = True
    component.append(vertex)
    
    for next_vertex in range(matrix.shape[0]):
        if matrix[vertex, next_vertex] == 1 and not visited[next_vertex]:
            dfs_component(next_vertex, matrix, visited, component)

def create_condensation_graph(matrix, components):
    """Create condensation graph from strongly connected components"""
    n_components = len(components)
    condensation_matrix = np.zeros((n_components, n_components), dtype=int)
    
    # Check for edges between components
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                # Check if there's an edge from any vertex in component i
                # to any vertex in component j
                for v1 in [v-1 for v in components[i]]:  # Convert to 0-indexed
                    for v2 in [v-1 for v in components[j]]:  # Convert to 0-indexed
                        if matrix[v1, v2] == 1:
                            condensation_matrix[i, j] = 1
                            break
                    if condensation_matrix[i, j] == 1:
                        break
    
    return condensation_matrix

def format_path(path):
    """Format a path for printing"""
    return " – ".join(str(v+1) for v in path)  # Convert to 1-indexed