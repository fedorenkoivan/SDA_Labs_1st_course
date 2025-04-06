import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.collections
import random

# Set variant parameters
variant = 4229
n1, n2, n3, n4 = 4, 2, 2, 9  # Extracted from variant number 4229
n = 10 + n3  # Number of vertices = 10 + 2 = 12
k = 1.0 - n3*0.02 - n4*0.005 - 0.25  # k = 1.0 - 2*0.02 - 9*0.005 - 0.25 = 0.455

def generate_adjacency_matrix(n, variant):
    """Generate directed graph adjacency matrix according to the algorithm."""
    np.random.seed(variant)
    
    # Generate random matrix and apply threshold in one step
    T = np.random.random((n, n)) * 2.0
    A = (T * k >= 1.0).astype(int)
    
    return A

def create_undirected_matrix(A_dir):
    """Create undirected graph adjacency matrix from directed graph matrix."""
    # Use logical OR and symmetrize in one operation
    return np.logical_or(A_dir, A_dir.T).astype(int)

def calculate_vertex_positions(n, layout_type):
    """Calculate vertex positions based on layout type."""
    positions = []
    
    if layout_type == "rectangle_center":
        # Place vertices in a rectangle with one in the center
        center = (0.5, 0.5)
        positions.append(center)  # Center vertex
        
        # Place remaining n-1 vertices on rectangle
        vertices_on_rectangle = n - 1
        
        # More rectangular shape
        rect_width, rect_height = 0.7, 0.4
        
        # Calculate positions on perimeter
        top_count = vertices_on_rectangle // 4 + (1 if vertices_on_rectangle % 4 > 0 else 0)
        right_count = vertices_on_rectangle // 4 + (1 if vertices_on_rectangle % 4 > 1 else 0)
        bottom_count = vertices_on_rectangle // 4 + (1 if vertices_on_rectangle % 4 > 2 else 0)
        left_count = vertices_on_rectangle - top_count - right_count - bottom_count
        
        # Top edge
        for i in range(top_count):
            x = 0.5 - rect_width/2 + rect_width * (i + 1) / (top_count + 1)
            y = 0.5 + rect_height/2
            positions.append((x, y))
        
        # Right edge
        for i in range(right_count):
            x = 0.5 + rect_width/2
            y = 0.5 + rect_height/2 - rect_height * (i + 1) / (right_count + 1)
            positions.append((x, y))
            
        # Bottom edge
        for i in range(bottom_count):
            x = 0.5 + rect_width/2 - rect_width * (i + 1) / (bottom_count + 1)
            y = 0.5 - rect_height/2
            positions.append((x, y))
            
        # Left edge
        for i in range(left_count):
            x = 0.5 - rect_width/2
            y = 0.5 - rect_height/2 + rect_height * (i + 1) / (left_count + 1)
            positions.append((x, y))
            
    return positions

def create_arrow(start_pos, end_pos, vertex_radius):
    """Create an arrow patch between two vertices."""
    # Calculate direction vector and normalize
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return None  # We'll handle self-loops separately
    
    # Normal arrow
    dx, dy = dx/length, dy/length
    
    # Adjust start and end points to be on the edge of the vertices
    start_x = start_pos[0] + dx * vertex_radius
    start_y = start_pos[1] + dy * vertex_radius
    end_x = end_pos[0] - dx * vertex_radius
    end_y = end_pos[1] - dy * vertex_radius
    
    return FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                         arrowstyle='->', mutation_scale=10, 
                         color='black', zorder=1)

def create_line(start_pos, end_pos, vertex_radius):
    """Create a line segment between two vertices."""
    # Calculate direction vector and normalize
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return None  # We'll handle self-loops separately
    
    # Normal line
    dx, dy = dx/length, dy/length
    
    # Adjust start and end points to be on the edge of the vertices
    start_x = start_pos[0] + dx * vertex_radius
    start_y = start_pos[1] + dy * vertex_radius
    end_x = end_pos[0] - dx * vertex_radius
    end_y = end_pos[1] - dy * vertex_radius
    
    return [(start_x, start_y), (end_x, end_y)]

def draw_graph(ax, A, positions, directed=True):
    """Draw a graph based on adjacency matrix and vertex positions."""
    n = len(A)
    vertex_radius = 0.03
    vertex_color = 'skyblue'
    vertex_border = 'black'
    
    # Create collections for batch rendering
    circles = []
    for i, pos in enumerate(positions):
        circles.append(Circle(pos, radius=vertex_radius, facecolor=vertex_color, 
                            edgecolor=vertex_border, zorder=2))
        
        # Set vertex numbers
        vertex_number = 12 if i == 0 else i
        ax.text(pos[0], pos[1], str(vertex_number), horizontalalignment='center', 
               verticalalignment='center', zorder=3)
    
    # Add all circles at once
    ax.add_collection(matplotlib.collections.PatchCollection(circles, match_original=True))
    
    # Handle self-loops separately
    for i in range(n):
        if A[i][i] == 1:
            center_x, center_y = positions[i]
            radius = vertex_radius * 1.5
            
            if directed:
                # Self-loop for directed graph
                arc = patches.Arc((center_x, center_y + radius), radius*2, radius*2, 
                                theta1=180, theta2=360, linewidth=1.0, fill=False, zorder=1)
                ax.add_patch(arc)
                
                # Add arrowhead
                arrow_x = center_x + radius * np.cos(np.pi)
                arrow_y = center_y + radius + radius * np.sin(np.pi)
                arrow = FancyArrowPatch((arrow_x, arrow_y + 0.01), (arrow_x, arrow_y), 
                                      arrowstyle='->', mutation_scale=10, color='black', zorder=1)
                ax.add_patch(arrow)
            else:
                # Self-loop for undirected graph
                circle = Circle((center_x, center_y + radius), radius, fill=False, 
                              edgecolor='black', zorder=1)
                ax.add_patch(circle)
    
    # Draw edges (excluding self-loops which are already drawn)
    if directed:
        arrows = []
        for i in range(n):
            for j in range(n):
                if A[i][j] == 1 and i != j:  # Skip self-loops
                    arrow = create_arrow(positions[i], positions[j], vertex_radius)
                    if arrow:
                        arrows.append(arrow)
        
        # Add all arrows at once
        for arrow in arrows:
            ax.add_patch(arrow)
    else:
        lines = []
        for i in range(n):
            for j in range(i+1, n):  # Only process upper triangle
                if A[i][j] == 1 and i != j:  # Skip self-loops
                    line = create_line(positions[i], positions[j], vertex_radius)
                    if line:
                        lines.append(line)
        
        # Add all lines at once
        if lines:
            ax.add_collection(matplotlib.collections.LineCollection(lines, colors='black', zorder=1))

def print_matrix(matrix, title):
    """Print a matrix in a nice format."""
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(str(val) for val in row))

def main():
    # Generate directed graph adjacency matrix
    A_dir = generate_adjacency_matrix(n, variant)
    print_matrix(A_dir, "Directed Graph Adjacency Matrix")
    
    # Generate undirected graph adjacency matrix from directed
    A_undir = create_undirected_matrix(A_dir)
    print_matrix(A_undir, "Undirected Graph Adjacency Matrix")
    
    # Calculate vertex positions
    positions = calculate_vertex_positions(n, "rectangle_center")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Add margin around the plots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Draw directed graph
    ax1.set_title("Directed Graph")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    draw_graph(ax1, A_dir, positions, directed=True)
    
    # Draw undirected graph
    ax2.set_title("Undirected Graph")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    draw_graph(ax2, A_undir, positions, directed=False)
    
    # Use tight_layout with padding
    plt.tight_layout(pad=3.0)
    
    # Save figure with high quality
    plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graph visualization completed successfully!")

if __name__ == "__main__":
    main()