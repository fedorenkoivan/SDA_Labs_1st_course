import shutil
import matplotlib.pyplot as plt
from matrix_methods import (
    generate_adjacency_matrix, get_undirected_matrix,
    calculate_reachability_matrix, calculate_strong_connectivity_matrix,
    print_matrix, calculate_degrees
)
from drawing_methods import (
    draw_graph, draw_condensation_graph,
    get_vertex_positions
)
from graph_analysis import (
    is_regular_graph, find_special_vertices,
    find_paths_of_length, find_strongly_connected_components,
    create_condensation_graph, format_path
)

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
    
    def print_paths_optimized(paths, paths_length_title):
        print(f"\n{paths_length_title} (count: {len(paths)}):")
        
        if not paths:
            print("Немає шляхів для відображення")
            return
        
        formatted_paths = [format_path(path) for path in paths]
        max_path_length = max(len(path) for path in formatted_paths) + 2
        
        terminal_width = shutil.get_terminal_size().columns
        
        max_columns = max(1, terminal_width // (max_path_length + 1))
        
        for i in range(0, len(paths), max_columns):
            row_paths = formatted_paths[i:i+max_columns]
            row_str = ""
            for j, path in enumerate(row_paths):
                if j > 0:
                    row_str += "| "
                row_str += f"{path:<{max_path_length-2}}"
            print(row_str)

    paths_length_2 = find_paths_of_length(new_directed_matrix, 2)
    print_paths_optimized(paths_length_2, "Paths of length 2")

    paths_length_3 = find_paths_of_length(new_directed_matrix, 3)
    print_paths_optimized(paths_length_3, "Paths of length 3")
    
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