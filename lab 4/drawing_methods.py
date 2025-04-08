import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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