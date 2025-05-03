import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.widgets import Button

class Vertex:
    def __init__(self, id, x=0, y=0):
        self.id = id
        self.x = x
        self.y = y
        self.adjacency_list = []  # List of (vertex_id, weight) tuples
    
    def add_adjacent(self, vertex_id, weight):
        # Add to adjacency list if not already there
        for i, (v, w) in enumerate(self.adjacency_list):
            if v == vertex_id:
                self.adjacency_list[i] = (vertex_id, weight)
                return
        self.adjacency_list.append((vertex_id, weight))
    
    def remove_adjacent(self, vertex_id):
        # Remove from adjacency list
        self.adjacency_list = [(v, w) for v, w in self.adjacency_list if v != vertex_id]

class Graph:
    def __init__(self):
        self.vertices = {}  # {vertex_id: vertex_object}
    
    def add_vertex(self, id, x=0, y=0):
        if id not in self.vertices:
            self.vertices[id] = Vertex(id, x, y)
        return self.vertices[id]
    
    def remove_vertex(self, id):
        if id in self.vertices:
            # Remove all edges connected to this vertex
            for v_id in self.vertices:
                if v_id != id:
                    self.vertices[v_id].remove_adjacent(id)
            # Remove the vertex itself
            del self.vertices[id]
    
    def add_edge(self, src, dest, weight):
        if src not in self.vertices:
            self.add_vertex(src)
        if dest not in self.vertices:
            self.add_vertex(dest)
        
        # Add to adjacency lists of both vertices (undirected graph)
        self.vertices[src].add_adjacent(dest, weight)
        self.vertices[dest].add_adjacent(src, weight)
    
    def remove_edge(self, src, dest):
        if src in self.vertices and dest in self.vertices:
            self.vertices[src].remove_adjacent(dest)
            self.vertices[dest].remove_adjacent(src)

def generate_adjacency_matrix(n, variant_number):
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    n3 = 2
    n4 = 9
    # Updated coefficient according to the new formula
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05  # Now k = 1.0 - 0.02 - 0.045 - 0.05 = 0.885
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 if T[i, j] * k >= 1.0 else 0
    return A, T

def get_undirected_matrix(directed_matrix):
    n = directed_matrix.shape[0]
    undirected_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if directed_matrix[i, j] == 1:
                undirected_matrix[i, j] = 1
                undirected_matrix[j, i] = 1
    return undirected_matrix

def generate_weight_matrix(undirected_matrix, random_matrix_B):
    n = undirected_matrix.shape[0]
    
    # Step 2: Calculate matrix C
    C = np.ceil(random_matrix_B * 100 * undirected_matrix)
    
    # Step 3: Calculate matrix D
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            D[i, j] = 1 if C[i, j] > 0 else 0
    
    # Step 4: Calculate matrix H
    H = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 if D[i, j] == D[j, i] else 0
    
    # Step 5: Create upper triangular matrix Tr
    Tr = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i <= j:  # Upper triangular including diagonal
                Tr[i, j] = 1
    
    # Step 6: Calculate weight matrix W
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            w_value = int(D[i, j] * H[i, j] * Tr[i, j] * C[i, j])
            W[i, j] = w_value
            W[j, i] = w_value  # Make it symmetric
    
    return W

def get_vertex_positions(n, n4):
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

def rotate_around_center(x, y, cx, cy, angle):
    """Rotate point (x, y) around center (cx, cy) by angle (in radians)"""
    dx = x - cx
    dy = y - cy
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_new = cx + dx * cos_a - dy * sin_a
    y_new = cy + dx * sin_a + dy * cos_a
    return (x_new, y_new)

def draw_self_loop(ax, pos, color='blue', linewidth=1.5, is_directed=False):
    """Draw a self-loop using polygonal lines under the vertex (with rotation)"""
    cx, cy = pos
    R = 0.5
    index_angle = 0
    theta = index_angle

    cx += R * math.sin(theta)
    cy -= R * math.cos(theta)

    dx = 3 * R / 4
    dy = R * (1 - math.sqrt(7)) / 4

    p1 = (cx - dx, cy - dy)
    p2 = (cx - 3 * dx / 2, cy - R / 2)
    p3 = (cx + 3 * dx / 2, cy - R / 2)
    p4 = (cx + dx, cy - dy)

    p1 = rotate_around_center(p1[0], p1[1], cx, cy, theta)
    p2 = rotate_around_center(p2[0], p2[1], cx, cy, theta)
    p3 = rotate_around_center(p3[0], p3[1], cx, cy, theta)
    p4 = rotate_around_center(p4[0], p4[1], cx, cy, theta)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=color, linewidth=linewidth)

    if is_directed:
        dx_arrow = p4[0] - p3[0]
        dy_arrow = p4[1] - p3[1]
        ax.arrow(p3[0], p3[1], dx_arrow, dy_arrow,
                 head_width=0.15, head_length=0.15,
                 fc=color, ec=color, linewidth=linewidth,
                 length_includes_head=True)
    else:
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color, linewidth=linewidth)

def draw_edge(ax, start, end, is_directed=False, color='blue', linewidth=1.5):
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
    rad = 0.0  # No curve for undirected graph
    
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->' if is_directed else '-',
        color=color,
        linewidth=linewidth,
        connectionstyle=f'arc3,rad={rad}',
        mutation_scale=15
    )
    ax.add_patch(arrow)

def matrix_to_graph(adj_matrix, weight_matrix, positions):
    n = adj_matrix.shape[0]
    graph = Graph()
    
    # Add vertices
    for i in range(n):
        graph.add_vertex(i+1, positions[i][0], positions[i][1])
    
    # Add edges
    for i in range(n):
        for j in range(i+1, n):  # Upper triangular to avoid duplicates
            if adj_matrix[i, j] == 1 and weight_matrix[i, j] > 0:
                graph.add_edge(i+1, j+1, weight_matrix[i, j])
    
    return graph

def prim_algorithm(graph, start_vertex=1):
    n = len(graph.vertices)
    if n == 0:
        return []
    
    mst_edges = []  # List to store MST edges (u, v, weight)
    mst_vertices = {start_vertex}  # Set of vertices in MST
    remaining_vertices = set(graph.vertices.keys()) - {start_vertex}
    
    while remaining_vertices:
        min_weight = float('inf')
        min_edge = None
        
        for u in mst_vertices:
            for v_id, weight in graph.vertices[u].adjacency_list:
                if v_id not in mst_vertices:
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (u, v_id, weight)
        
        if min_edge:
            u, v, weight = min_edge
            mst_edges.append(min_edge)
            mst_vertices.add(v)
            remaining_vertices.remove(v)
        else:
            break  # Graph is not connected
            
    return mst_edges

def step_by_step_prim(graph, start_vertex=1):
    n = len(graph.vertices)
    if n == 0:
        return []
    
    mst_edges = []  # List to store MST edges (u, v, weight)
    mst_vertices = {start_vertex}  # Set of vertices in MST
    remaining_vertices = set(graph.vertices.keys()) - {start_vertex}
    
    # Initial state - no edges yet
    yield mst_edges.copy()
    
    while remaining_vertices:
        min_weight = float('inf')
        min_edge = None
        
        for u in mst_vertices:
            for v_id, weight in graph.vertices[u].adjacency_list:
                if v_id not in mst_vertices:
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (u, v_id, weight)
        
        if min_edge:
            u, v, weight = min_edge
            mst_edges.append(min_edge)
            mst_vertices.add(v)
            remaining_vertices.remove(v)
            yield mst_edges.copy()
        else:
            break  # Graph is not connected

def draw_graph_with_mst(graph, mst_edges=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Draw all edges
    for u_id in graph.vertices:
        u = graph.vertices[u_id]
        for v_id, weight in u.adjacency_list:
            if u_id < v_id:  # To avoid drawing edges twice
                v = graph.vertices[v_id]
                u_pos = (u.x, u.y)
                v_pos = (v.x, v.y)
                
                # Check if this edge is in MST
                edge_in_mst = mst_edges and any((a == u_id and b == v_id) or (a == v_id and b == u_id) 
                                               for a, b, _ in mst_edges)
                color = 'red' if edge_in_mst else 'blue'
                linewidth = 2.5 if edge_in_mst else 1.0
                
                draw_edge(ax, u_pos, v_pos, is_directed=False, color=color, linewidth=linewidth)
                
                # Draw edge weight
                mid_x = (u_pos[0] + v_pos[0]) / 2
                mid_y = (u_pos[1] + v_pos[1]) / 2
                ax.text(mid_x, mid_y, str(weight), fontsize=9, horizontalalignment='center', 
                       verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw vertices
    for v_id in graph.vertices:
        v = graph.vertices[v_id]
        is_in_mst = mst_edges and any((a == v_id or b == v_id) for a, b, _ in mst_edges)
        color = 'lightgreen' if is_in_mst else 'lightblue'
        edge_color = 'green' if is_in_mst else 'blue'
        circle = plt.Circle((v.x, v.y), 0.5, fill=True, color=color, edgecolor=edge_color)
        ax.add_patch(circle)
        ax.text(v.x, v.y, str(v_id), horizontalalignment='center', 
                verticalalignment='center', fontsize=10, color='black', fontweight='bold')
    
    ax.set_aspect('equal')
    margin = 2
    
    # Calculate plot limits
    x_coords = [graph.vertices[v].x for v in graph.vertices]
    y_coords = [graph.vertices[v].y for v in graph.vertices]
    
    if x_coords and y_coords:  # Check if there are any vertices
        ax.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
        ax.set_ylim(min(y_coords)-margin, max(y_coords)+margin)
    
    if mst_edges:
        # Calculate total MST weight
        total_weight = sum(weight for _, _, weight in mst_edges)
        plt.title(f"MST Construction - Total Weight: {total_weight}")
    else:
        plt.title("Initial Graph")
    
    plt.axis('off')
    return fig, ax

class StepVisualizer:
    def __init__(self, graph, start_vertex=1):
        self.graph = graph
        self.mst_generator = step_by_step_prim(graph, start_vertex)
        self.current_mst = next(self.mst_generator)
        self.step = 0
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.update_figure()
        
        # Add buttons
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_step)
        self.btn_next.on_clicked(self.next_step)
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def update_figure(self):
        self.ax.clear()
        rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
        self.ax.add_patch(rect)
        
        # Draw all edges
        for u_id in self.graph.vertices:
            u = self.graph.vertices[u_id]
            for v_id, weight in u.adjacency_list:
                if u_id < v_id:  # To avoid drawing edges twice
                    v = self.graph.vertices[v_id]
                    u_pos = (u.x, u.y)
                    v_pos = (v.x, v.y)
                    
                    # Check if this edge is in MST
                    edge_in_mst = any((a == u_id and b == v_id) or (a == v_id and b == u_id) 
                                    for a, b, _ in self.current_mst)
                    color = 'red' if edge_in_mst else 'blue'
                    linewidth = 2.5 if edge_in_mst else 1.0
                    
                    draw_edge(self.ax, u_pos, v_pos, is_directed=False, color=color, linewidth=linewidth)
                    
                    # Draw edge weight
                    mid_x = (u_pos[0] + v_pos[0]) / 2
                    mid_y = (u_pos[1] + v_pos[1]) / 2
                    self.ax.text(mid_x, mid_y, str(weight), fontsize=9, horizontalalignment='center', 
                               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Draw vertices
        for v_id in self.graph.vertices:
            v = self.graph.vertices[v_id]
            is_in_mst = any((a == v_id or b == v_id) for a, b, _ in self.current_mst)
            color = 'lightgreen' if is_in_mst else 'lightblue'
            edge_color = 'green' if is_in_mst else 'blue'
            circle = plt.Circle((v.x, v.y), 0.5, fill=True, color=color, edgecolor=edge_color)
            self.ax.add_patch(circle)
            self.ax.text(v.x, v.y, str(v_id), horizontalalignment='center', 
                    verticalalignment='center', fontsize=10, color='black', fontweight='bold')
        
        self.ax.set_aspect('equal')
        margin = 2
        
        # Calculate plot limits
        x_coords = [self.graph.vertices[v].x for v in self.graph.vertices]
        y_coords = [self.graph.vertices[v].y for v in self.graph.vertices]
        
        if x_coords and y_coords:
            self.ax.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
            self.ax.set_ylim(min(y_coords)-margin, max(y_coords)+margin)
        
        if self.current_mst:
            total_weight = sum(weight for _, _, weight in self.current_mst)
            self.ax.set_title(f"MST Construction - Step {self.step} - Total Weight: {total_weight}")
        else:
            self.ax.set_title("Initial Graph - Step 0")
        
        self.ax.axis('off')
        plt.draw()
    
    def next_step(self, event=None):
        try:
            self.current_mst = next(self.mst_generator)
            self.step += 1
            self.update_figure()
        except StopIteration:
            print("MST construction complete!")
    
    def prev_step(self, event=None):
        if self.step > 0:
            # Restart the generator and advance to the previous step
            self.mst_generator = step_by_step_prim(self.graph, 1)
            self.current_mst = next(self.mst_generator)  # Initial state
            self.step = 0
            
            for _ in range(self.step):
                try:
                    self.current_mst = next(self.mst_generator)
                    self.step += 1
                except StopIteration:
                    break
            
            self.update_figure()
    
    def on_key(self, event):
        if event.key == 'right' or event.key == ' ':
            self.next_step()
        elif event.key == 'left':
            self.prev_step()

def print_matrix(matrix, title):
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def main():
    variant_number = 4229
    n1, n2, n3, n4 = 4, 2, 2, 9
    n = 10 + n3  # n = 12
    
    print(f"Variant number: {variant_number}")
    print(f"n1={n1}, n2={n2}, n3={n3}, n4={n4}")
    print(f"Number of vertices n = 10 + {n3} = {n}")
    print(f"Vertex placement: Rectangular with vertex in center (n4 = {n4})")
    
    # Generate adjacency matrices
    directed_matrix, random_matrix = generate_adjacency_matrix(n, variant_number)
    undirected_matrix = get_undirected_matrix(directed_matrix)
    
    # Generate weight matrix for the undirected graph
    np.random.seed(variant_number)  # Reset seed for consistent random numbers
    random_matrix_B = np.random.random((n, n)) * 2.0
    weight_matrix = generate_weight_matrix(undirected_matrix, random_matrix_B)
    
    print_matrix(directed_matrix, f"Directed Graph Adjacency Matrix ({n}x{n})")
    print_matrix(undirected_matrix, f"Undirected Graph Adjacency Matrix ({n}x{n})")
    print_matrix(weight_matrix, f"Weight Matrix ({n}x{n})")
    
    # Get vertex positions
    positions = get_vertex_positions(n, n4)
    
    # Create graph from matrices
    graph = matrix_to_graph(undirected_matrix, weight_matrix, positions)
    
    # Find MST using Prim's algorithm
    mst_edges = prim_algorithm(graph)
    print("\nMinimum Spanning Tree Edges:")
    total_weight = 0
    for u, v, weight in mst_edges:
        total_weight += weight
        print(f"Edge ({u}, {v}) with weight {weight}")
    print(f"Total MST weight: {total_weight}")
    
    # Visualize graph and MST step by step
    print("\nStarting step-by-step visualization...")
    print("Press SPACE or RIGHT ARROW to advance, LEFT ARROW to go back")
    visualizer = StepVisualizer(graph)
    plt.show()

if __name__ == "__main__":
    main()