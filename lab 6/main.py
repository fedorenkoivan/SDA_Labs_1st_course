import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --- Прямокутне розташування як на малюнку ---
def rectangle_with_center_layout():
    """Return coordinates for 12 nodes: 4 top, 3 right, 3 bottom, 2 left, 1 center (1-indexed)."""
    coords = {}
    # Rectangle dimensions
    W, H = 8, 6
    mx, my = 1.2, 1
    # Top row (1-4)
    for i in range(4):
        coords[i+1] = (mx + i*(W-2*mx)/3, H-my)
    # Right column (5-7)
    for i in range(3):
        coords[5+i] = (W-mx, H-my-(i+1)*(H-2*my)/4)
    # Bottom row (8-10), right to left
    for i in range(3):
        coords[8+i] = (W-mx-(i+1)*(W-2*mx)/3, my)
    # Left column (11-12), bottom to top
    coords[11] = (mx, my+(H-2*my)/4)
    coords[12] = (mx, my+2*(H-2*my)/4)
    # Center (13, but our vertex 12!)
    coords[0] = (W/2, H/2)
    return coords

# --- Генерація графа ---
n1, n2, n3, n4 = 4, 2, 2, 9
variant = int(f"{n1}{n2}{n3}{n4}")
n = 12

np.random.seed(variant)
Adir = np.random.uniform(0, 2.0, (n, n))
k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05
Adir = Adir * k
Adir = np.where(Adir < 1.0, 0, 1)
Aundir = np.logical_or(Adir, Adir.T).astype(int)

np.random.seed(variant)
B = np.random.uniform(0, 2.0, (n, n))
C = np.ceil(B * 100 * Aundir).astype(int)
D = np.where(C == 0, 0, 1)
H = np.where(D == D.T, 1, 0)
Tr = np.triu(np.ones((n, n)), 1)
W = D * H * Tr * C
W = W + W.T  # Symmetric

def matrix_to_adjlist(W):
    n = W.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0:
                adj[i].append((j, int(W[i, j])))
    return adj
adj = matrix_to_adjlist(W)

# --- Прямокутне розташування (vertex 0 — центр, далі 1...11 по колу) ---
layout = rectangle_with_center_layout()
# Перевірка: вершини з 1 по 12, 0 — центр
node_pos = {i: layout[i] for i in range(1, 13)}
node_pos[12] = layout[0]  # 12 - центр

# --- Реалізація Пріма з покроковим виконанням ---
class PrimStepVisualizer:
    def __init__(self, adj):
        self.n = len(adj)
        self.adj = adj
        self.reset()
    def reset(self):
        self.in_mst = [False]*self.n
        self.edge_to = [-1]*self.n
        self.weight_to = [float('inf')]*self.n
        self.weight_to[0] = 0
        self.heap = [(0, 0)]
        self.mst_edges = []
        self.last_added = None
    def step(self):
        while self.heap:
            self.heap.sort()
            w, u = self.heap.pop(0)
            if self.in_mst[u]:
                continue
            self.in_mst[u] = True
            if self.edge_to[u] != -1:
                self.mst_edges.append((self.edge_to[u], u))
                self.last_added = (self.edge_to[u], u)
            else:
                self.last_added = None
            for v, weight in self.adj[u]:
                if not self.in_mst[v] and weight < self.weight_to[v]:
                    self.weight_to[v] = weight
                    self.edge_to[v] = u
                    self.heap.append((weight, v))
            break
    def get_mst_edges(self):
        return list(self.mst_edges)
    def is_done(self):
        return all(self.in_mst)

def draw_graph(adj, W, pos, mst_edges, last_added):
    plt.clf()
    # Всі ребра
    for u in range(len(adj)):
        for v, w in adj[u]:
            if u < v:
                color = "black"
                width = 1
                if (u, v) in mst_edges or (v, u) in mst_edges:
                    color = "red"
                    width = 2.5
                if last_added and ((u, v) == last_added or (v, u) == last_added):
                    color = "orange"
                    width = 4
                x0, y0 = pos[u+1 if u != 11 else 12]
                x1, y1 = pos[v+1 if v != 11 else 12]
                plt.plot([x0, x1], [y0, y1], color=color, lw=width, zorder=1)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                plt.text(mx, my, str(int(W[u, v])), color="blue", fontsize=9, ha='center', va='center',
                         bbox=dict(facecolor="white", edgecolor='none', pad=0.1), zorder=2)
    # Вершини
    for i in range(1, 13):
        x, y = pos[i]
        plt.scatter([x], [y], s=450, color='lightblue', edgecolor='k', zorder=3)
        plt.text(x, y, str(i), color="black", fontsize=14, ha='center', va='center', fontweight='bold', zorder=4)
    plt.title("Алгоритм Пріма: кістяк графа")
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.pause(0.05)

class StepApp:
    def __init__(self, adj, W, pos):
        self.prim = PrimStepVisualizer(adj)
        self.adj = adj
        self.W = W
        self.pos = pos
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn = Button(self.button_ax, 'Наступний крок')
        self.btn.on_clicked(self.next_step)
        self.reset_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.btn_reset = Button(self.reset_ax, 'Скинути')
        self.btn_reset.on_clicked(self.reset)
        self.show_step()
    def show_step(self):
        draw_graph(self.adj, self.W, self.pos, self.prim.get_mst_edges(), self.prim.last_added)
        if self.prim.is_done():
            self.btn.label.set_text('Готово')
            self.btn.ax.set_visible(False)
        else:
            self.btn.label.set_text('Наступний крок')
            self.btn.ax.set_visible(True)
    def next_step(self, event):
        if not self.prim.is_done():
            self.prim.step()
            self.show_step()
    def reset(self, event):
        self.prim.reset()
        self.btn.label.set_text('Наступний крок')
        self.btn.ax.set_visible(True)
        self.show_step()
    def show(self):
        plt.show()

if __name__ == "__main__":
    app = StepApp(adj, W, node_pos)
    app.show()