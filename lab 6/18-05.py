import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import sys

# --- PART 1: Generation of the required matrices ---

def generate_adjacency_matrix(n, variant_number, n3, n4):
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            val = T[i, j] * k
            A[i, j] = 1 if val >= 1.0 else 0
    return A

def get_undirected_matrix(directed_matrix):
    n = directed_matrix.shape[0]
    undirected_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if directed_matrix[i, j] == 1 or directed_matrix[j, i] == 1:
                undirected_matrix[i, j] = 1
                undirected_matrix[j, i] = 1
    return undirected_matrix

def generate_weight_matrix(undirected_matrix, n, variant_number):
    np.random.seed(variant_number)
    B = np.random.random((n, n)) * 2.0
    # Step 2: C = ceil(b_ij*100*aund_ij)
    C = np.ceil(B * 100 * undirected_matrix).astype(int)
    # Step 3: D = 0 if c_ij==0 else 1
    D = (C > 0).astype(int)
    # Step 4: H = 1 if d_ij==d_ji else 0
    H = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 if D[i, j] == D[j, i] else 0
    # Step 5: Tr (upper triangle including diag is 0, under is 1)
    Tr = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i > j:
                Tr[i, j] = 1
    # Step 6: w_ij = w_ji = d_ij * h_ij * tr_ij * c_ij
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            W[i, j] = D[i, j] * H[i, j] * Tr[i, j] * C[i, j]
            W[j, i] = W[i, j]  # Ensure symmetry
    return W

# --- PART 2: Graph visualization (rectangular + center) ---

def get_vertex_positions(n, n4):
    positions = np.zeros((n, 2))
    if n4 in [8, 9]:
        width, height = 12, 8
        positions[n-1] = [0, 0]
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
    else:
        # fallback: circle
        for i in range(n):
            angle = 2 * np.pi * i / n
            positions[i] = [6 * np.cos(angle), 4 * np.sin(angle)]
    return positions

def draw_edge(ax, start, end, color='blue', linewidth=1.5, weight=None, weight_pos=0.5, highlight=False):
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
    ax.plot([start_x, end_x], [start_y, end_y],
            color=color if not highlight else "red",
            linewidth=linewidth if not highlight else 3)
    # Draw the weight
    if weight is not None and weight > 0:
        wx = start_x + (end_x - start_x) * weight_pos
        wy = start_y + (end_y - start_y) * weight_pos
        ax.text(wx, wy, str(weight), fontsize=9, color=color if not highlight else "red", weight="bold")

def draw_graph(W, positions, mst_edges=None, highlight_color="red"):
    n = W.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))
    rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    # draw all edges
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0:
                is_mst = mst_edges is not None and ((i, j) in mst_edges or (j, i) in mst_edges)
                draw_edge(ax, positions[i], positions[j], color='blue', linewidth=1.5, weight=W[i, j], highlight=is_mst)
    # draw nodes
    for i, pos in enumerate(positions):
        circle = plt.Circle(pos, 0.5, fill=True, color='lightblue', edgecolor='blue')
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i+1), horizontalalignment='center',
                verticalalignment='center', fontsize=10, color='black', fontweight='bold')
    ax.set_aspect('equal')
    margin = 2
    ax.set_xlim(min(positions[:, 0])-margin, max(positions[:, 0])+margin)
    ax.set_ylim(min(positions[:, 1])-margin, max(positions[:, 1])+margin)
    plt.axis('off')
    return fig, ax

# --- PART 3: Dynamic Graph Structure (for MST algorithms) ---

class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight
    def __lt__(self, other):
        return self.weight < other.weight

class Graph:
    def __init__(self, n):
        self.n = n
        self.edges = []  # (u, v, weight)
        self.adj = [[] for _ in range(n)]
    def add_edge(self, u, v, weight):
        self.edges.append(Edge(u, v, weight))
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
    def get_edges(self):
        return self.edges
    def get_neighbors(self, u):
        return self.adj[u]

def build_graph_from_matrix(W):
    n = W.shape[0]
    g = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0:
                g.add_edge(i, j, W[i, j])
    return g

# --- PART 4: Kruskal's and Prim's algorithms with step-by-step visualization ---

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        xroot, yroot = self.find(x), self.find(y)
        if xroot == yroot:
            return False
        self.parent[yroot] = xroot
        return True

def kruskal_stepwise(graph, positions, W):
    n = graph.n
    edges = sorted(graph.get_edges(), key=lambda e: e.weight)
    ds = DisjointSet(n)
    mst_edges = []
    print("\n=== Kruskal's Algorithm Step-by-Step ===")
    for idx, edge in enumerate(edges):
        print(f"\nStep {idx+1}: Considering edge ({edge.u+1}, {edge.v+1}) with weight {edge.weight}")
        if ds.union(edge.u, edge.v):
            mst_edges.append((edge.u, edge.v))
            print("Added to MST.")
        else:
            print("Edge forms a cycle. Skipped.")
        fig, ax = draw_graph(W, positions, mst_edges)
        plt.title(f"Kruskal's Algorithm Step {idx+1}")
        plt.show(block=False)
        input("Press Enter for next step...")
        plt.close()
        if len(mst_edges) == n-1:
            break
    print("MST edges:", [(u+1, v+1) for u, v in mst_edges])
    return mst_edges

def prim_stepwise(graph, positions, W):
    n = graph.n
    visited = [False]*n
    mst_edges = []
    import heapq
    heap = []
    visited[0] = True
    for v, w in graph.get_neighbors(0):
        heapq.heappush(heap, (w, 0, v))
    print("\n=== Prim's Algorithm Step-by-Step ===")
    step_count = 1
    while heap and len(mst_edges) < n-1:
        w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        mst_edges.append((u, v))
        print(f"\nStep {step_count}: Added edge ({u+1}, {v+1}) with weight {w}")
        visited[v] = True
        for to, ww in graph.get_neighbors(v):
            if not visited[to]:
                heapq.heappush(heap, (ww, v, to))
        fig, ax = draw_graph(W, positions, mst_edges)
        plt.title(f"Prim's Algorithm Step {step_count}")
        plt.show(block=False)
        input("Press Enter for next step...")
        plt.close()
        step_count += 1
    print("MST edges:", [(u+1, v+1) for u, v in mst_edges])
    return mst_edges

# --- PART 5: Main program and user interaction ---

def print_matrix(matrix, title):
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(map(str, row)))

def main():
    # --- Parameters for your variant ---
    variant_number = 4229
    n3 = 2
    n4 = 9
    n = 10 + n3
    print(f"Variant number: {variant_number}")
    print(f"Number of vertices n = 10 + {n3} = {n}")
    print(f"Vertex placement: Rectangular with vertex in center (n4 = {n4})")

    # --- Generate matrices ---
    directed_matrix = generate_adjacency_matrix(n, variant_number, n3, n4)
    undirected_matrix = get_undirected_matrix(directed_matrix)
    W = generate_weight_matrix(undirected_matrix, n, variant_number)
    positions = get_vertex_positions(n, n4)

    print_matrix(directed_matrix, f"Directed Graph Adjacency Matrix ({n}x{n})")
    print_matrix(undirected_matrix, f"Undirected Graph Adjacency Matrix ({n}x{n})")
    print_matrix(W, f"Weight Matrix W ({n}x{n})")

    # --- Draw the undirected weighted graph ---
    fig, ax = draw_graph(W, positions)
    plt.title("Undirected Weighted Graph")
    plt.show()

    # --- Build dynamic graph for MST ---
    graph = build_graph_from_matrix(W)

    # --- Choose and run MST algorithm step-by-step ---
    if n4 % 2 == 0:
        print("\n n4 is even, using Kruskal's algorithm.")
        kruskal_stepwise(graph, positions, W)
    else:
        print("\n n4 is odd, using Prim's algorithm.")
        prim_stepwise(graph, positions, W)

if __name__ == "__main__":
    main()