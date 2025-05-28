import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def rectangle_with_center_layout():
    coords = {}
    W, H = 8, 6
    mx, my = 1.2, 1
    for i in range(4):
        coords[i+1] = (mx + i*(W-2*mx)/3, H-my)
    for i in range(3):
        coords[5+i] = (W-mx, H-my-(i+1)*(H-2*my)/4)
    for i in range(3):
        coords[8+i] = (W-mx-(i+1)*(W-2*mx)/3, my)
    coords[11] = (mx, my+(H-2*my)/4)
    coords[12] = (mx, my+2*(H-2*my)/4)
    coords[0] = (W/2, H/2)
    return coords

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
W = W + W.T

def matrix_to_adjlist(W):
    n = W.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0:
                adj[i].append((j, int(W[i, j])))
    return adj
adj = matrix_to_adjlist(W)

layout = rectangle_with_center_layout()
node_pos = {}
for i in range(1, 13):
    if i == 12:
        node_pos[i] = layout[0]
    else:
        node_pos[i] = layout[i]

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

class StepApp:
    def __init__(self, adj, W, pos):
        self.prim = PrimStepVisualizer(adj)
        self.adj = adj
        self.W = W
        self.pos = pos
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.22)
        self.button_ax = plt.axes([0.32, 0.05, 0.36, 0.1])
        self.btn = Button(self.button_ax, 'Наступний крок', color='#007bff', hovercolor='#0056b3')
        self.btn.on_clicked(self.next_step)
        self.btn_active = True
        self.reset_ax = plt.axes([0.78, 0.05, 0.18, 0.1])
        self.btn_reset = Button(self.reset_ax, 'Скинути', color='#f0ad4e', hovercolor='#ec971f')
        self.btn_reset.on_clicked(self.reset)
        self.redraw()
    def redraw(self):
        self.ax.clear()
        # Draw all edges
        for u in range(len(self.adj)):
            for v, w in self.adj[u]:
                if u < v:
                    color = "black"
                    width = 1
                    if (u, v) in self.prim.mst_edges or (v, u) in self.prim.mst_edges:
                        color = "red"
                        width = 2.5
                    if self.prim.last_added and ((u, v) == self.prim.last_added or (v, u) == self.prim.last_added):
                        color = "orange"
                        width = 4
                    ix = 12 if u == 11 else u+1
                    jx = 12 if v == 11 else v+1
                    x0, y0 = self.pos[ix]
                    x1, y1 = self.pos[jx]
                    self.ax.plot([x0, x1], [y0, y1], color=color, lw=width, zorder=1)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    self.ax.text(mx, my, str(int(self.W[u, v])), color="blue", fontsize=9, ha='center', va='center',
                                 bbox=dict(facecolor="white", edgecolor='none', pad=0.1), zorder=2)
        # Draw nodes
        for i in range(1, 13):
            x, y = self.pos[i]
            self.ax.scatter([x], [y], s=450, color='lightblue', edgecolor='k', zorder=3)
            self.ax.text(x, y, str(i), color="black", fontsize=14, ha='center', va='center', fontweight='bold', zorder=4)
        self.ax.set_title("Алгоритм Пріма: кістяк графа")
        self.ax.axis('off')
        self.ax.axis('equal')
        if self.prim.is_done():
            self.btn.label.set_text('Готово')
            self.btn.color = '#cccccc'
            self.btn.hovercolor = '#cccccc'
            self.btn_active = False
        else:
            self.btn.label.set_text('Наступний крок')
            self.btn.color = '#007bff'
            self.btn.hovercolor = '#0056b3'
            self.btn_active = True
        self.fig.canvas.draw_idle()
    def next_step(self, event):
        if self.btn_active and not self.prim.is_done():
            self.prim.step()
            self.redraw()
    def reset(self, event):
        self.prim.reset()
        self.btn.label.set_text('Наступний крок')
        self.btn.color = '#007bff'
        self.btn.hovercolor = '#0056b3'
        self.btn_active = True
        self.redraw()
    def show(self):
        plt.show()

if __name__ == "__main__":
    app = StepApp(adj, W, node_pos)
    app.show()