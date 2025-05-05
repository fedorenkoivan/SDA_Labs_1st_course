import matplotlib.pyplot as plt
import numpy as np
from matrix_methods import generate_adjacency_matrix, print_matrix
from drawing_methods import (
    draw_graph,
    get_vertex_positions
)
from traversal_methods import draw_traversal_graph

def main():
    variant_number = 4229
    n3 = 2
    n4 = 9
    n = 10 + n3
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.15

    print(f"Variant number: {variant_number}")
    print(f"Number of vertices n = 10 + {n3} = {n}")
    
    # PART 5: Graph Traversal - New for Laboratory Work #5
    print("\n=== PART 5: Graph Traversal Analysis ===")
    
    # Generate matrix using existing function with correct coefficient
    directed_matrix = generate_adjacency_matrix(n, variant_number, k)
    
    print_matrix(directed_matrix, f"Directed Graph Adjacency Matrix ({n}x{n}) for Traversal")
    
    # Get vertex positions
    positions = get_vertex_positions(n, n4)
    
    # Draw original graph before traversal
    fig = draw_graph(directed_matrix, positions, is_directed=True, title="Graph for Traversal")
    fig.savefig('graph_for_traversal.png')
    
    # Perform BFS traversal with visualization
    print("\nStarting Breadth-First Search (BFS)...")
    print("Press any key in plot window to continue step by step")
    bfs_fig, bfs_forest, bfs_order = draw_traversal_graph(directed_matrix, positions, "BFS")
    bfs_fig.savefig('bfs_traversal.png')
    
    # Perform DFS traversal with visualization
    print("\nStarting Depth-First Search (DFS)...")
    print("Press any key in plot window to continue step by step")
    dfs_fig, dfs_forest, dfs_order = draw_traversal_graph(directed_matrix, positions, "DFS")
    dfs_fig.savefig('dfs_traversal.png')
    
    # Analyze and compare traversals
    print("\nTraversal Analysis:")
    print(f"BFS visited {len(bfs_order)} vertices")
    print(f"DFS visited {len(dfs_order)} vertices")
    
    # BFS tree properties
    bfs_tree_edges = sum(len(tree) for tree in bfs_forest)
    print(f"BFS created {len(bfs_forest)} tree(s) with a total of {bfs_tree_edges} edges")
    
    # DFS tree properties  
    dfs_tree_edges = sum(len(tree) for tree in dfs_forest)
    print(f"DFS created {len(dfs_forest)} tree(s) with a total of {dfs_tree_edges} edges")
    
    plt.show()

if __name__ == "__main__":
    main()