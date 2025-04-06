import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import random

def generate_adjacency_matrix(n, variant_number):
    """Generate adjacency matrix according to the requirements"""
    # Set random seed based on variant number
    np.random.seed(variant_number)
    
    # Generate random matrix with values in [0, 2.0)
    T = np.random.random((n, n)) * 2.0
    
    # Calculate coefficient k
    n3 = (variant_number // 10) % 10
    n4 = variant_number % 10
    k = 1.0 - n3 * 0.02 - n4 * 0.005 - 0.25
    print(f"Coefficient k = {k}")
    
    # Multiply matrix by coefficient and round
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 if T[i, j] * k >= 1.0 else 0
    
    return A

def get_undirected_matrix(dir_matrix):
    """Convert directed adjacency matrix to undirected"""
    n = dir_matrix.shape[0]
    undir_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if dir_matrix[i, j] == 1:
                undir_matrix[i, j] = 1
                undir_matrix[j, i] = 1
    
    return undir_matrix

def get_vertex_positions(n, n4):
    """Get vertex positions based on n4 value"""
    if n4 in [8, 9]:  # Rectangle with a vertex in the center
        # Rectangle dimensions
        width, height = 12, 8
        
        # Initialize the positions array
        positions = np.zeros((n, 2))
        
        # Set the center vertex (the last one)
        positions[n-1] = [0, 0]
        
        # The first n-1 vertices will be on the perimeter of the rectangle
        perimeter_vertices = n - 1  # 11 in our case
        
        # We need to place vertices on a perfect rectangle
        # Calculate how many vertices go on each side of the rectangle
        # For a perfect rectangle, we need to place them at the corners and along the sides
        
        # Need at least 4 vertices for the corners
        corners = [(width/2, height/2), (width/2, -height/2), (-width/2, -height/2), (-width/2, height/2)]
        
        # Remaining vertices to place on the sides
        remaining = perimeter_vertices - 4
        
        # Calculate vertices per side (excluding corners)
        # Distribute remaining vertices evenly (prioritizing top/bottom if needed)
        sides = [0, 0, 0, 0]  # top, right, bottom, left
        
        # First, distribute evenly
        for i in range(remaining):
            sides[i % 4] += 1
        
        # Create positions array
        index = 0
        
        # Add top-left corner
        positions[index] = [-width/2, height/2]
        index += 1
        
        # Add top side (excluding corners)
        for i in range(sides[0]):
            x = -width/2 + (i+1) * width / (sides[0]+1)
            positions[index] = [x, height/2]
            index += 1
        
        # Add top-right corner
        positions[index] = [width/2, height/2]
        index += 1
        
        # Add right side (excluding corners)
        for i in range(sides[1]):
            y = height/2 - (i+1) * height / (sides[1]+1)
            positions[index] = [width/2, y]
            index += 1
        
        # Add bottom-right corner
        positions[index] = [width/2, -height/2]
        index += 1
        
        # Add bottom side (excluding corners)
        for i in range(sides[2]):
            x = width/2 - (i+1) * width / (sides[2]+1)
            positions[index] = [x, -height/2]
            index += 1
        
        # Add bottom-left corner
        positions[index] = [-width/2, -height/2]
        index += 1
        
        # Add left side (excluding corners)
        for i in range(sides[3]):
            y = -height/2 + (i+1) * height / (sides[3]+1)
            positions[index] = [-width/2, y]
            index += 1
        
        return positions
    
    return np.zeros((n, 2))  # Default empty positions

def draw_curved_arrow(ax, start, end, color='k', linewidth=1.5, rad=0.3):
    """Draw a curved arrow from start to end"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Calculate the distance between points
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:  # Avoid division by zero
        return
        
    # Calculate the coordinates for the arrow to not overlap with vertices
    # Assuming vertex radius is 0.5
    vertex_radius = 0.5
    ratio = vertex_radius / dist
    start_x = start[0] + dx * ratio
    start_y = start[1] + dy * ratio
    end_x = end[0] - dx * ratio
    end_y = end[1] - dy * ratio
    
    # Make arrow heads more visible with larger arrowstyle
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->',
        color=color,
        linewidth=linewidth,
        connectionstyle=f'arc3,rad={rad}',
        mutation_scale=15  # Makes arrow heads larger
    )
    ax.add_patch(arrow)

def draw_curved_line(ax, start, end, color='k', linewidth=1.5, rad=0.3):
    """Draw a curved line from start to end without an arrow"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Calculate the distance between points
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.001:  # Avoid division by zero
        return
        
    # Calculate the coordinates for the line to not overlap with vertices
    vertex_radius = 0.5
    ratio = vertex_radius / dist
    start_x = start[0] + dx * ratio
    start_y = start[1] + dy * ratio
    end_x = end[0] - dx * ratio
    end_y = end[1] - dy * ratio
    
    # Create a curved line without an arrow
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='-',  # No arrow, just a line
        color=color,
        linewidth=linewidth,
        connectionstyle=f'arc3,rad={rad}'
    )
    ax.add_patch(arrow)

def draw_graph(adjacency_matrix, positions, is_directed=True):
    """Draw a graph based on adjacency matrix and vertex positions"""
    n = adjacency_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw rectangle outline to visualize the alignment
    rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Draw edges
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] == 1:
                # Determine curvature based on whether there's a reciprocal edge
                has_reciprocal = adjacency_matrix[j, i] == 1 and i != j
                rad = 0.3 if has_reciprocal else 0.1
                
                if is_directed:
                    # Use a different color for directed edges to make them more visible
                    draw_curved_arrow(ax, positions[i], positions[j], rad=rad, color='red', linewidth=1.8)
                else:
                    # For undirected graph, draw each edge only once
                    if i <= j:  # This avoids drawing the same edge twice
                        draw_curved_line(ax, positions[i], positions[j], rad=rad, color='blue', linewidth=1.5)
    
    # Draw vertices
    for i, pos in enumerate(positions):
        circle = plt.Circle(pos, 0.5, fill=True, color='lightblue', edgecolor='blue')
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i+1), horizontalalignment='center', 
                verticalalignment='center', fontsize=10, color='black', fontweight='bold')
    
    # Set axis properties
    ax.set_aspect('equal')
    margin = 2
    ax.set_xlim(min(positions[:, 0])-margin, max(positions[:, 0])+margin)
    ax.set_ylim(min(positions[:, 1])-margin, max(positions[:, 1])+margin)
    
    graph_type = "Directed" if is_directed else "Undirected"
    plt.title(f"{graph_type} Graph - {n} vertices")
    
    # Add a legend to clarify the meaning of arrows
    if is_directed:
        arrow_legend = patches.FancyArrowPatch((0, 0), (1, 0), 
                                               arrowstyle='->', color='red',
                                               linewidth=1.8, 
                                               connectionstyle='arc3,rad=0')
        ax.legend([arrow_legend], ['Direction of edge'], loc='upper right')
    
    plt.axis('off')
    
    return fig, ax

def print_matrix(matrix, title):
    """Print the adjacency matrix in a readable format"""
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def main():
    # Parameters based on variant 4229
    variant_number = 4229
    n3 = 2  # (variant_number // 10) % 10
    n4 = 9  # variant_number % 10
    n = 10 + n3  # Number of vertices = 12
    
    print(f"Variant number: {variant_number}")
    print(f"n3 = {n3}, n4 = {n4}")
    print(f"Number of vertices n = 10 + {n3} = {n}")
    print(f"Vertex placement: Rectangular with vertex in center (n4 = {n4})")
    
    # Generate directed adjacency matrix
    dir_matrix = generate_adjacency_matrix(n, variant_number)
    
    # Generate undirected adjacency matrix
    undir_matrix = get_undirected_matrix(dir_matrix)
    
    # Get vertex positions
    positions = get_vertex_positions(n, n4)
    
    # Print matrices
    print_matrix(dir_matrix, f"Directed Graph Adjacency Matrix ({n}x{n})")
    print_matrix(undir_matrix, f"Undirected Graph Adjacency Matrix ({n}x{n})")
    
    # Print vertex positions for verification
    print("\nVertex positions:")
    for i, pos in enumerate(positions):
        print(f"Vertex {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    # Draw directed graph
    print("\nCreating directed graph with arrows...")
    draw_graph(dir_matrix, positions, is_directed=True)
    plt.savefig('directed_graph.png')
    
    # Draw undirected graph
    print("Creating undirected graph without arrows...")
    draw_graph(undir_matrix, positions, is_directed=False)
    plt.savefig('undirected_graph.png')
    
    plt.show()

if __name__ == "__main__":
    main()