import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_adjacency_matrix(n, variant_number, k_formula=1):
    """Generate directed adjacency matrix based on variant number and k_formula
    k_formula=1: k = 1.0 - n3 * 0.01 - n4 * 0.01 - 0.3
    k_formula=2: k = 1.0 - n3 * 0.005 - n4 * 0.005 - 0.27
    """
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    
    n3 = 2  # From variant
    n4 = 9  # From variant
    
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
    
    # Calculate matrix power
    power_matrix = np.linalg.matrix_power(matrix, length)
    
    # Find all paths by looking at the power matrix entries
    paths = []
    for i in range(n):
        for j in range(n):
            if power_matrix[i, j] > 0:
                # We need to find all the actual paths from i to j
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

def calculate_strong_connectivity_matrix(reachability):
    """Calculate strong connectivity matrix from reachability matrix"""
    n = reachability.shape[0]
    strong_connectivity = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if reachability[i, j] == 1 and reachability[j, i] == 1:
                strong_connectivity[i, j] = 1
    
    return strong_connectivity

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

def get_vertex_positions(n, n4):
    """Get vertex positions based on n4 value"""
    positions = np.zeros((n, 2))
    
    if n4 in [8, 9]:
        width, height = 12, 8
        positions[n-1] = [0, 0]  # Center vertex
        
        perimeter_vertices = n - 1
        sides = [0, 0, 0, 0]
        remaining = perimeter_vertices - 4
        
        for i in range(remaining):
            sides[i % 4] += 1
        
        vertex_index = 0
        
        positions[vertex_index] = [-width/2, height/2]
        vertex_index += 1
        
        for i in range(sides[0]):
            x = -width/2 + (i+1) * width / (sides[0]+1)
            positions[vertex_index] = [x, height/2]
            vertex_index += 1
        
        positions[vertex_index] = [width/2, height/2]
        vertex_index += 1
        
        for i in range(sides[1]):
            y = height/2 - (i+1) * height / (sides[1]+1)
            positions[vertex_index] = [width/2, y]
            vertex_index += 1
        
        positions[vertex_index] = [width/2, -height/2]
        vertex_index += 1
        
        for i in range(sides[2]):
            x = width/2 - (i+1) * width / (sides[2]+1)
            positions[vertex_index] = [x, -height/2]
            vertex_index += 1
        
        positions[vertex_index] = [-width/2, -height/2]
        vertex_index += 1
        
        for i in range(sides[3]):
            y = -height/2 + (i+1) * height / (sides[3]+1)
            positions[vertex_index] = [-width/2, y]
            vertex_index += 1
    
    return positions

def get_component_positions(components):
    """Create positions for components in condensation graph"""
    n_components = len(components)
    radius = 5
    
    positions = []
    for i in range(n_components):
        angle = 2 * np.pi * i / n_components
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append([x, y])
    
    return np.array(positions)

def draw_edge(ax, start, end, is_directed=True, color='blue', linewidth=1.5):
    """Draw an edge between two vertices"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:
        return
    
    vertex_radius = 0.5
    ratio = vertex_radius / dist
    start_x = start[0] + dx * ratio
    start_y = start[1] + dy * ratio
    end_x = end[0] - dx * ratio
    end_y = end[1] - dy * ratio
    
    rad = 0.2
    
    if is_directed:
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->',
            color=color,
            linewidth=linewidth,
            connectionstyle=f'arc3,rad={rad}',
            mutation_scale=15
        )
    else:
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='-',
            color=color,
            linewidth=linewidth,
            connectionstyle=f'arc3,rad={rad}'
        )
    
    ax.add_patch(arrow)

def draw_graph(adjacency_matrix, positions, is_directed=True, title="Graph"):
    """Draw a graph based on adjacency matrix and vertex positions"""
    n = adjacency_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', 
                            facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] == 1:
                draw_edge(ax, positions[i], positions[j], is_directed, color='blue')
    
    for i, pos in enumerate(positions):
        circle = plt.Circle(pos, 0.5, fill=True, color='lightblue', edgecolor='blue')
        ax.add_patch(circle)
        
        ax.text(pos[0], pos[1], str(i+1), horizontalalignment='center', 
                verticalalignment='center', fontsize=10, color='black', fontweight='bold')
    
    ax.set_aspect('equal')
    margin = 2
    ax.set_xlim(min(positions[:, 0])-margin, max(positions[:, 0])+margin)
    ax.set_ylim(min(positions[:, 1])-margin, max(positions[:, 1])+margin)
    
    graph_type = "Directed" if is_directed else "Undirected"
    plt.title(f"{title} - {graph_type} Graph - {n} vertices")
    
    plt.axis('off')
    return fig

def draw_condensation_graph(condensation_matrix, components, positions=None):
    """Draw the condensation graph"""
    n_components = len(components)
    
    if positions is None:
        positions = get_component_positions(components)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i in range(n_components):
        for j in range(n_components):
            if condensation_matrix[i, j] == 1:
                draw_edge(ax, positions[i], positions[j], is_directed=True, color='red')
    
    for i, pos in enumerate(positions):
        radius = 0.8  # Larger radius for component nodes
        circle = plt.Circle(pos, radius, fill=True, color='lightgreen', edgecolor='green')
        ax.add_patch(circle)
        
        # Label with component number and list of vertices
        component_label = f"C{i+1}: {components[i]}"
        ax.text(pos[0], pos[1], component_label, horizontalalignment='center', 
                verticalalignment='center', fontsize=9, color='black')
    
    ax.set_aspect('equal')
    margin = 3
    ax.set_xlim(min(positions[:, 0])-margin, max(positions[:, 0])+margin)
    ax.set_ylim(min(positions[:, 1])-margin, max(positions[:, 1])+margin)
    
    plt.title(f"Condensation Graph - {n_components} components")
    plt.axis('off')
    return fig

def print_matrix(matrix, title):
    """Print the adjacency matrix in a readable format"""
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def format_path(path):
    """Format a path for printing"""
    return " – ".join(str(v+1) for v in path)  # Convert to 1-indexed

def main():
    variant_number = 4229
    n3 = 2
    n4 = 9
    n = 10 + n3
    
    print(f"Variant number: {variant_number}")
    print(f"Number of vertices n = 10 + {n3} = {n}")
    
    print("\n=== PART 1: Original Graph Analysis ===")
    directed_matrix = generate_adjacency_matrix(n, variant_number, k_formula=1)
    undirected_matrix = get_undirected_matrix(directed_matrix)
    
    print_matrix(directed_matrix, f"Directed Graph Adjacency Matrix ({n}x{n})")
    print_matrix(undirected_matrix, f"Undirected Graph Adjacency Matrix ({n}x{n})")
    
    in_degrees, out_degrees = calculate_degrees(directed_matrix, is_directed=True)
    undirected_degrees = calculate_degrees(undirected_matrix, is_directed=False)
    
    print("\nDirected Graph:")
    print("Vertex | In-degree | Out-degree")
    for i in range(n):
        print(f"{i+1:6d} | {in_degrees[i]:9d} | {out_degrees[i]:10d}")
    
    print("\nUndirected Graph:")
    print("Vertex | Degree")
    for i in range(n):
        print(f"{i+1:6d} | {undirected_degrees[i]:6d}")
    
    if is_regular_graph(undirected_degrees):
        print(f"\nUndirected graph is regular with degree {undirected_degrees[0]}")
    else:
        print("\nUndirected graph is not regular")
    
    if is_regular_graph(in_degrees) and is_regular_graph(out_degrees) and in_degrees[0] == out_degrees[0]:
        print(f"Directed graph is regular with in-degree = out-degree = {in_degrees[0]}")
    else:
        print("Directed graph is not regular")
    
    # Find hanging and isolated vertices
    dir_hanging, dir_isolated = find_special_vertices(directed_matrix, is_directed=True)
    undir_hanging, undir_isolated = find_special_vertices(undirected_matrix, is_directed=False)
    
    print("\nDirected Graph:")
    print(f"Hanging vertices: {dir_hanging}")
    print(f"Isolated vertices: {dir_isolated}")
    
    print("\nUndirected Graph:")
    print(f"Hanging vertices: {undir_hanging}")
    print(f"Isolated vertices: {undir_isolated}")
    
    # Draw graphs
    positions = get_vertex_positions(n, n4)
    fig1 = draw_graph(directed_matrix, positions, is_directed=True, title="Original")
    fig1.savefig('directed_graph_original.png')
    
    fig2 = draw_graph(directed_matrix, positions, is_directed=False, title="Original")
    fig2.savefig('undirected_graph_original.png')
    
    print("\n\n=== PART 2: Modified Graph Analysis ===")
    new_directed_matrix = generate_adjacency_matrix(n, variant_number, k_formula=2)
    print_matrix(new_directed_matrix, f"Modified Directed Graph Adjacency Matrix ({n}x{n})")
    
    new_in_degrees, new_out_degrees = calculate_degrees(new_directed_matrix, is_directed=True)
    
    print("\nModified Directed Graph:")
    print("Vertex | In-degree | Out-degree")
    for i in range(n):
        print(f"{i+1:6d} | {new_in_degrees[i]:9d} | {new_out_degrees[i]:10d}")
    
    paths_length_2 = find_paths_of_length(new_directed_matrix, 2)
    print(f"\nPaths of length 2 (count: {len(paths_length_2)}):")
    for path in paths_length_2:
        print(format_path(path))
    
    paths_length_3 = find_paths_of_length(new_directed_matrix, 3)
    print(f"\nPaths of length 3 (count: {len(paths_length_3)}):")
    for path in paths_length_3:
        print(format_path(path))
    
    reachability_matrix = calculate_reachability_matrix(new_directed_matrix)
    print_matrix(reachability_matrix, "Reachability Matrix")
    
    strong_connectivity_matrix = calculate_strong_connectivity_matrix(reachability_matrix)
    print_matrix(strong_connectivity_matrix, "Strong Connectivity Matrix")
    
    components = find_strongly_connected_components(strong_connectivity_matrix)
    print("\nStrongly Connected Components:")
    for i, component in enumerate(components):
        print(f"Component {i+1}: {component}")
    
    condensation_matrix = create_condensation_graph(new_directed_matrix, components)
    print_matrix(condensation_matrix, "Condensation Graph Adjacency Matrix")
    
    fig3 = draw_graph(new_directed_matrix, positions, is_directed=True, title="Modified")
    fig3.savefig('directed_graph_modified.png')
    
    fig4 = draw_condensation_graph(condensation_matrix, components)
    fig4.savefig('condensation_graph.png')
    
    plt.show()

if __name__ == "__main__":
    main()