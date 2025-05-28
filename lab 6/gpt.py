import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Генерація матриці суміжності
def generate_adjacency_matrix(n, variant_number, n3, n4):
    np.random.seed(variant_number)
    T = np.random.random((n, n)) * 2.0
    k = 1.0 - n3 * 0.01 - n4 * 0.005 - 0.05
    A = (T * k >= 1.0).astype(int)
    return A

# Симетризація
def get_undirected_matrix(A):
    return np.maximum(A, A.T)

# Матриця ваг
def generate_weight_matrix(Aundir, B):
    C = np.ceil(B * 100 * Aundir).astype(int)
    D = (C > 0).astype(int)
    H = (D == D.T).astype(int)
    Tr = np.triu(np.ones_like(Aundir), k=0)
    W = D * H * Tr * C
    W = W + W.T - np.diag(W.diagonal())
    return W

# Розташування вершин у прямокутнику з центром
def get_vertex_positions(n, n4):
    positions = np.zeros((n, 2))
    if n4 in [8, 9]:
        width, height = 12, 8
        positions[n - 1] = [0, 0]
        perimeter = n - 1
        sides = [0, 0, 0, 0]
        for i in range(perimeter - 4):
            sides[i % 4] += 1
        idx = 0
        positions[idx] = [-width / 2, height / 2]; idx += 1
        for i in range(sides[0]):
            positions[idx] = [-width / 2 + (i + 1) * width / (sides[0] + 1), height / 2]; idx += 1
        positions[idx] = [width / 2, height / 2]; idx += 1
        for i in range(sides[1]):
            positions[idx] = [width / 2, height / 2 - (i + 1) * height / (sides[1] + 1)]; idx += 1
        positions[idx] = [width / 2, -height / 2]; idx += 1
        for i in range(sides[2]):
            positions[idx] = [width / 2 - (i + 1) * width / (sides[2] + 1), -height / 2]; idx += 1
        positions[idx] = [-width / 2, -height / 2]; idx += 1
        for i in range(sides[3]):
            positions[idx] = [-width / 2, -height / 2 + (i + 1) * height / (sides[3] + 1)]; idx += 1
    return positions

# Малювання графа
def draw_graph(Aundir, W, positions):
    n = Aundir.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))
    rect = patches.Rectangle((-6, -4), 12, 8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Ребра з вагами
    for i in range(n):
        for j in range(i + 1, n):
            if Aundir[i, j] and W[i, j] > 0:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=1.5)

                # Обчислимо зміщення для ваги
                dx = x2 - x1
                dy = y2 - y1
                offset_x = -0.2 * dy / np.sqrt(dx**2 + dy**2 + 1e-9)
                offset_y =  0.2 * dx / np.sqrt(dx**2 + dy**2 + 1e-9)
                mx = (x1 + x2) / 2 + offset_x
                my = (y1 + y2) / 2 + offset_y

                ax.text(mx, my, str(W[i, j]), fontsize=9, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

    # Вершини
    for i in range(n):
        x, y = positions[i]
        ax.add_patch(plt.Circle((x, y), 0.4, color='lightblue', ec='black'))
        ax.text(x, y, str(i + 1), ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_title("Undirected Weighted Graph (Variant 4229)")
    ax.axis('equal')
    ax.axis('off')
    plt.show()


# === Запуск ===
variant = 4229
n1, n2, n3, n4 = 4, 2, 2, 9
n = 10 + n3

A = generate_adjacency_matrix(n, variant, n3, n4)
Aundir = get_undirected_matrix(A)
np.random.seed(variant)
B = np.random.random((n, n)) * 2.0
W = generate_weight_matrix(Aundir, B)
positions = get_vertex_positions(n, n4)
draw_graph(Aundir, W, positions)
