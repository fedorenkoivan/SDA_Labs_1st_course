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
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05  # k = 1.0 - 0.02 - 0.045 - 0.05 = 0.885
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
    # НАДІЙНА кругова розкладка
    positions = np.zeros((n, 2))
    radius = 5
    angle_step = 2 * np.pi / n
    for i in range(n):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[i] = [x, y]
    return positions

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
    """
    Find minimum spanning tree using Prim's algorithm
    """
    if not graph.vertices:
        return []
    
    mst_edges = []
    mst_vertices = {start_vertex}
    remaining_vertices = set(graph.vertices.keys()) - {start_vertex}
    
    while remaining_vertices:
        min_weight = float('inf')
        min_edge = None
        
        # Find minimum weight edge connecting MST to remaining vertices
        for u in mst_vertices:
            for v_id, weight in graph.vertices[u].adjacency_list:
                if v_id not in mst_vertices and weight < min_weight:
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

def generate_all_steps(graph, start_vertex=1):
    """
    Generate all steps of Prim's algorithm as a list for easy navigation
    """
    all_steps = []
    mst_edges = []
    mst_vertices = {start_vertex}
    remaining_vertices = set(graph.vertices.keys()) - {start_vertex}
    
    # Initial state - no edges yet
    all_steps.append([])
    
    while remaining_vertices:
        min_weight = float('inf')
        min_edge = None
        
        for u in mst_vertices:
            for v_id, weight in graph.vertices[u].adjacency_list:
                if v_id not in mst_vertices and weight < min_weight:
                    min_weight = weight
                    min_edge = (u, v_id, weight)
        
        if min_edge:
            u, v, weight = min_edge
            mst_edges.append(min_edge)
            mst_vertices.add(v)
            remaining_vertices.remove(v)
            all_steps.append(mst_edges.copy())
        else:
            break  # Graph is not connected
    
    return all_steps

class StepVisualizer:
    def __init__(self, graph, start_vertex=1):
        self.graph = graph
        self.step_index = 0
        self.all_steps = generate_all_steps(graph, start_vertex)  # ← ваш реальний MST крок за кроком

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)

        ax_prev = plt.axes([0.3, 0.05, 0.15, 0.06])
        ax_next = plt.axes([0.55, 0.05, 0.15, 0.06])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')

        self.btn_prev.on_clicked(self.prev_step)
        self.btn_next.on_clicked(self.next_step)

        self.update_figure()

    def update_figure(self):
        self.ax.clear()
        current_edges = self.all_steps[self.step_index]

        # Draw edges
        for u_id in self.graph.vertices:
            u = self.graph.vertices[u_id]
            for v_id, weight in u.adjacency_list:
                if u_id < v_id:
                    v = self.graph.vertices[v_id]
                    is_mst = (u_id, v_id, weight) in current_edges or (v_id, u_id, weight) in current_edges
                    color = 'red' if is_mst else 'gray'
                    self.ax.plot([u.x, v.x], [u.y, v.y], color=color, linewidth=2)

                    mx, my = (u.x + v.x) / 2, (u.y + v.y) / 2
                    self.ax.text(mx, my, str(weight), fontsize=9,
                                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        # Draw vertices
        mst_vertices = set()
        for u, v, _ in current_edges:
            mst_vertices.add(u)
            mst_vertices.add(v)

        for v_id in self.graph.vertices:
            v = self.graph.vertices[v_id]
            color = 'lightgreen' if v_id in mst_vertices else 'lightblue'
            edge_color = 'green' if v_id in mst_vertices else 'black'
            self.ax.add_patch(plt.Circle((v.x, v.y), 0.4, color=color, ec=edge_color, lw=2))
            self.ax.text(v.x, v.y, str(v_id), ha='center', va='center', fontsize=12, fontweight='bold')

        self.ax.set_title(f"MST Construction - Step {self.step_index} / {len(self.all_steps) - 1}")
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def next_step(self, event=None):
        if self.step_index < len(self.all_steps) - 1:
            self.step_index += 1
            self.update_figure()

    def prev_step(self, event=None):
        if self.step_index > 0:
            self.step_index -= 1
            self.update_figure()

def print_matrix(matrix, title):
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def main():
    variant_number = 4229
    n1, n2, n3, n4 = 4, 2, 2, 9
    n = 10 + n3  # = 12

    print(f"Variant: {variant_number}, n = {n}, n4 = {n4}")

    # 1. Генеруємо матриці
    directed_matrix, random_matrix = generate_adjacency_matrix(n, variant_number)
    undirected_matrix = get_undirected_matrix(directed_matrix)

    np.random.seed(variant_number)
    random_matrix_B = np.random.random((n, n)) * 2.0
    weight_matrix = generate_weight_matrix(undirected_matrix, random_matrix_B)

    # 2. Виводимо матриці для діагностики
    print_matrix(directed_matrix, "Directed Matrix")
    print_matrix(undirected_matrix, "Undirected Matrix")
    print_matrix(weight_matrix, "Weight Matrix")

    # 3. Генеруємо позиції (по колу!)
    positions = get_vertex_positions(n, n4)

    # 4. Будуємо граф
    graph = matrix_to_graph(undirected_matrix, weight_matrix, positions)

    # 5. Діагностика графа
    print(f"\nGraph has {len(graph.vertices)} vertices:")
    for v_id, vertex in graph.vertices.items():
        print(f"Vertex {v_id}: {vertex.adjacency_list}")

    # 6. Перевіримо MST
    mst_edges = prim_algorithm(graph)
    print("\nMST edges:")
    for u, v, w in mst_edges:
        print(f"{u} — {v} (weight {w})")

    if not mst_edges:
        print("\n⚠️ MST не знайдено — граф може бути незв’язним або вага всіх ребер = 0!")

    # 7. Запускаємо візуалізатор
    visualizer = StepVisualizer(graph)
    plt.show(block=True)

if __name__ == "__main__":
    main()