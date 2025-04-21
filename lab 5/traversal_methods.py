import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.patches import Patch
from drawing_methods import draw_edge, draw_self_loop

def find_start_vertex(adjacency_matrix):
    """Find the vertex with smallest index that has at least one outgoing edge"""
    n = adjacency_matrix.shape[0]
    for i in range(n):
        if np.sum(adjacency_matrix[i]) > 0:
            return i
    return 0  # Default to vertex 0 if no vertices have outgoing edges

def bfs(adjacency_matrix, start_vertex=None, ax=None, positions=None,
        vertex_circles=None, edge_arrows=None, pause_time=1.0):
    """Performs a breadth-first search traversal of the graph."""
    n = adjacency_matrix.shape[0]
    
    if start_vertex is None:
        start_vertex = find_start_vertex(adjacency_matrix)
    
    visited = [False] * n
    queue = deque([start_vertex])
    visited[start_vertex] = True
    parent = {start_vertex: None}
    traversal_edges = []
    traversal_order = []
    
    if ax is not None and positions is not None and vertex_circles is not None:
        vertex_circles[start_vertex].set_facecolor('yellow')
        ax.text(positions[start_vertex][0], positions[start_vertex][1] - 0.8, f"Start", 
                horizontalalignment='center', verticalalignment='center', fontsize=8)
        plt.pause(pause_time)
    
    while queue:
        current = queue.popleft()
        traversal_order.append(current)
        
        if ax is not None and positions is not None and vertex_circles is not None:
            vertex_circles[current].set_facecolor('red')
            plt.pause(pause_time)
        
        for neighbor in range(n):
            if adjacency_matrix[current, neighbor] == 1 and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
                traversal_edges.append((current, neighbor))
                
                if ax is not None and positions is not None and vertex_circles is not None and edge_arrows is not None:
                    vertex_circles[neighbor].set_facecolor('yellow')
                    edge_key = (current, neighbor)
                    if edge_key in edge_arrows:
                        edge_arrows[edge_key].set_color('green')
                        edge_arrows[edge_key].set_linewidth(2.5)
                    plt.pause(pause_time)
        
        if ax is not None and positions is not None and vertex_circles is not None:
            vertex_circles[current].set_facecolor('lightgreen')
            plt.pause(pause_time)
    
    return visited, parent, traversal_edges, traversal_order

def dfs(adjacency_matrix, start_vertex=None, ax=None, positions=None,
        vertex_circles=None, edge_arrows=None, pause_time=1.0):
    """Performs a depth-first search traversal of the graph."""
    n = adjacency_matrix.shape[0]
    
    if start_vertex is None:
        start_vertex = find_start_vertex(adjacency_matrix)
    
    visited = [False] * n
    parent = {start_vertex: None}
    traversal_edges = []
    traversal_order = []
    
    if ax is not None and positions is not None and vertex_circles is not None:
        vertex_circles[start_vertex].set_facecolor('yellow')
        ax.text(positions[start_vertex][0], positions[start_vertex][1] - 0.8, f"Start", 
                horizontalalignment='center', verticalalignment='center', fontsize=8)
        plt.pause(pause_time)
    
    def dfs_recursive(current):
        traversal_order.append(current)
        visited[current] = True
        
        if ax is not None and positions is not None and vertex_circles is not None:
            vertex_circles[current].set_facecolor('red')
            plt.pause(pause_time)
        
        for neighbor in range(n):
            if adjacency_matrix[current, neighbor] == 1 and not visited[neighbor]:
                parent[neighbor] = current
                traversal_edges.append((current, neighbor))
                
                # Update visualization for neighbor and edge
                if ax is not None and positions is not None and vertex_circles is not None and edge_arrows is not None:
                    edge_key = (current, neighbor)
                    if edge_key in edge_arrows:
                        edge_arrows[edge_key].set_color('green')
                        edge_arrows[edge_key].set_linewidth(2.5)
                    vertex_circles[neighbor].set_facecolor('yellow')
                    plt.pause(pause_time)
                
                dfs_recursive(neighbor)
        
        # Update visualization for visited vertex (processed)
        if ax is not None and positions is not None and vertex_circles is not None:
            vertex_circles[current].set_facecolor('lightgreen')
            plt.pause(pause_time)
    
    # Start DFS from the designated vertex
    dfs_recursive(start_vertex)
    
    return visited, parent, traversal_edges, traversal_order

def complete_graph_traversal(adjacency_matrix, traversal_func, ax=None, positions=None,
                            vertex_circles=None, edge_arrows=None, pause_time=1.0):
    """Completes a full traversal of the graph, handling disconnected components."""
    n = adjacency_matrix.shape[0]
    all_visited = [False] * n
    forest = []
    all_traversal_order = []
    
    while False in all_visited:
        # Find the smallest unvisited vertex with outgoing edges
        start_vertex = None
        for i in range(n):
            if not all_visited[i] and np.sum(adjacency_matrix[i]) > 0:
                start_vertex = i
                break
        
        # If no unvisited vertices with outgoing edges, choose any unvisited vertex
        if start_vertex is None:
            for i in range(n):
                if not all_visited[i]:
                    start_vertex = i
                    break
        
        # Run traversal from start vertex
        visited, parent, traversal_edges, traversal_order = traversal_func(
            adjacency_matrix, start_vertex, ax, positions, vertex_circles, edge_arrows, pause_time
        )
        
        # Update all_visited
        for i in range(n):
            all_visited[i] = all_visited[i] or visited[i]
        
        # Add traversal tree to forest
        if traversal_edges:
            forest.append(traversal_edges)
        
        all_traversal_order.extend(traversal_order)
    
    return forest, all_traversal_order

def draw_traversal_graph(adjacency_matrix, positions, traversal_func_name, pause_time=1.0):
    """Draw graph and perform traversal visualization step by step"""
    n = adjacency_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    vertex_circles = {}
    edge_arrows = {}
    
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] == 1:
                if i == j:
                    draw_self_loop(ax, positions[i], color='blue', linewidth=1.5, is_directed=True)
                else:
                    draw_edge(ax, positions[i], positions[j], is_directed=True, color='blue')
    
    for i in range(n):
        circle = plt.Circle(positions[i], 0.5, fill=True, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(circle)
        vertex_circles[i] = circle
        
        ax.text(positions[i][0], positions[i][1], str(i+1), horizontalalignment='center', 
                verticalalignment='center', fontsize=10, color='black', fontweight='bold')
    
    ax.set_aspect('equal')
    margin = 2
    ax.set_xlim(min(positions[:, 0])-margin, max(positions[:, 0])+margin)
    ax.set_ylim(min(positions[:, 1])-margin, max(positions[:, 1])+margin)
    
    plt.title(f"Graph Traversal - {traversal_func_name}")
    plt.axis('off')
    
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='Unvisited'),
        Patch(facecolor='yellow', edgecolor='blue', label='Discovered'),
        Patch(facecolor='red', edgecolor='blue', label='Processing'),
        Patch(facecolor='lightgreen', edgecolor='blue', label='Processed')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.pause(1.0)
    
    if traversal_func_name == "BFS":
        traversal_func = bfs
    else:  # "DFS"
        traversal_func = dfs
    
    forest, traversal_order = complete_graph_traversal(
        adjacency_matrix, traversal_func, ax, positions, vertex_circles, edge_arrows, pause_time
    )
    
    print(f"\n{traversal_func_name} Traversal Order:", " -> ".join(str(v+1) for v in traversal_order))
    
    print("\nTraversal Tree Edges:")
    for i, tree in enumerate(forest):
        print(f"Tree {i+1}:")
        for edge in tree:
            print(f"  {edge[0]+1} -> {edge[1]+1}")
    
    plt.title(f"Graph Traversal - {traversal_func_name} (Completed)")
    
    return fig, forest, traversal_order