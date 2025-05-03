import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import heapq

def generate_adjacency_matrix(n, variant_number, n3, n4):
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05
    A = (T * k >= 1.0).astype(int)
    return A

def get_undirected_matrix(A):
    return np.where((A + A.T) > 0, 1, 0)

def generate_weight_matrix(undir_matrix, n, variant_number):
    np.random.seed(variant_number)
    B = np.random.random((n, n)) * 2.0
    C = np.ceil(B * 100 * undir_matrix).astype(int)
    D = np.where(C > 0, 1, 0)
    H = np.where(D == D.T, 1, 0)
    Tr = np.triu(np.ones((n, n)), k=0)
    W = D * H * Tr * C
    W = W + W.T - np.diag(W.diagonal())
    return W

def get_rectangular_vertex_positions(n, n4):
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
        positions[vertex_index] = [-width/2, height/2]; vertex_index += 1
        for i in range(sides[0]):
            positions[vertex_index] = [-width/2 + (i+1) * width / (sides[0]+1), height/2]
            vertex_index += 1
        positions[vertex_index] = [width/2, height/2]; vertex_index += 1
        for i in range(sides[1]):
            positions[vertex_index] = [width/2, height/2 - (i+1) * height / (sides[1]+1)]
            vertex_index += 1
        positions[vertex_index] = [width/2, -height/2]; vertex_index += 1
        for i in range(sides[2]):
            positions[vertex_index] = [width/2 - (i+1) * width / (sides[2]+1), -height/2]
            vertex_index += 1
        positions[vertex_index] = [-width/2, -height/2]; vertex_index += 1
        for i in range(sides[3]):
            positions[vertex_index] = [-width/2, -height/2 + (i+1) * height / (sides[3]+1)]
            vertex_index += 1
    return positions

def draw_graph(W, positions, mst_edges=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    n = W.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], color='gray', linewidth=1)
                mid = (positions[i] + positions[j]) / 2
                ax.text(mid[0], mid[1], str(W[i, j]), color='black', fontsize=8)
    if mst_edges:
        for u, v, w in mst_edges:
            ax.plot([positions[u][0], positions[v][0]], [positions[u][1], positions[v][1]], color='red', linewidth=2)
    for i, pos in enumerate(positions):
        circle = plt.Circle(pos, 0.6, fill=True, color='lightblue', edgecolor='blue')
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i+1), ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Graph with MST Highlighted (red)")
    return fig, ax

def prim_algorithm(W):
    n = W.shape[0]
    visited = [False] * n
    mst_edges = []
    min_heap = [(0, 0, -1)]
    while min_heap and len(mst_edges) < n - 1:
        weight, u, parent = heapq.heappop(min_heap)
        if visited[u]: continue
        visited[u] = True
        if parent != -1:
            mst_edges.append((parent, u, weight))
        for v in range(n):
            if W[u][v] > 0 and not visited[v]:
                heapq.heappush(min_heap, (W[u][v], v, u))
    return mst_edges

class StepThroughMST:
    def __init__(self, W, positions, mst_steps):
        self.W = W
        self.positions = positions
        self.mst_steps = mst_steps
        self.current_edges = []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.2)
        self.button_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Next Step')
        self.button.on_clicked(self.next_step)
        self.draw_graph()

    def draw_graph(self):
        self.ax.clear()
        n = self.W.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if self.W[i, j] > 0:
                    self.ax.plot([self.positions[i][0], self.positions[j][0]], 
                                 [self.positions[i][1], self.positions[j][1]], 
                                 color='gray', linewidth=1)
                    mid = (self.positions[i] + self.positions[j]) / 2
                    self.ax.text(mid[0], mid[1], str(self.W[i, j]), color='black', fontsize=8)
        for u, v, w in self.current_edges:
            self.ax.plot([self.positions[u][0], self.positions[v][0]], 
                         [self.positions[u][1], self.positions[v][1]], 
                         color='red', linewidth=2)
        for i, pos in enumerate(self.positions):
            circle = plt.Circle(pos, 0.6, fill=True, color='lightblue', edgecolor='blue')
            self.ax.add_patch(circle)
            self.ax.text(pos[0], pos[1], str(i+1), ha='center', va='center', fontsize=12, fontweight='bold')
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title(f"Step {len(self.current_edges)} of {len(self.mst_steps)}")
        self.fig.canvas.draw()

    def next_step(self, event):
        if len(self.current_edges) < len(self.mst_steps):
            self.current_edges.append(self.mst_steps[len(self.current_edges)])
            self.draw_graph()

if __name__ == "__main__":
    variant_number = 4229
    n1, n2, n3, n4 = 4, 2, 2, 9
    n = 10 + n3
    Adir = generate_adjacency_matrix(n, variant_number, n3, n4)
    Aundir = get_undirected_matrix(Adir)
    W = generate_weight_matrix(Aundir, n, variant_number)
    positions = get_rectangular_vertex_positions(n, n4)
    mst = prim_algorithm(W)
    viewer = StepThroughMST(W, positions, mst)
    plt.show()