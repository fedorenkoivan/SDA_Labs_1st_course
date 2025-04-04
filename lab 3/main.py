import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Rectangle
import math

class GraphVisualizer:
    def __init__(self):
        self.n1 = 4
        self.n2 = 2
        self.n3 = 2
        self.n4 = 9
        
        self.n = 10 + self.n3
        
        self.k = 1.0 - self.n3 * 0.02 - self.n4 * 0.005 - 0.25

    def generate_adjacency_matrix(self):
        np.random.seed(self.n1 * 1000 + self.n2 * 100 + self.n3 * 10 + self.n4)
        
        T = np.random.uniform(0, 2.0, size=(self.n, self.n))
        
        A_dir = np.where(T * self.k >= 1.0, 1, 0)
        
        A_undir = np.maximum(A_dir, A_dir.T)
        
        return A_dir, A_undir

    def get_vertex_positions(self):
        positions = {}
        if self.n4 in [8, 9]:
            width, height = 8, 6
            outer_vertices = self.n - 1
            
            vertices_top = math.ceil(outer_vertices / 4)
            vertices_right = math.ceil((outer_vertices - vertices_top) / 3)
            vertices_bottom = math.ceil((outer_vertices - vertices_top - vertices_right) / 2)
            vertices_left = outer_vertices - vertices_top - vertices_right - vertices_bottom
            
            current_vertex = 0
            
            for i in range(vertices_top):
                if current_vertex >= outer_vertices: break
                x = -width/2 + i * (width/(vertices_top-1) if vertices_top > 1 else 0)
                positions[current_vertex] = (x, height/2)
                current_vertex += 1
            
            for i in range(vertices_right):
                if current_vertex >= outer_vertices: break
                y = height/2 - (i+1) * (height/(vertices_right+1))
                positions[current_vertex] = (width/2, y)
                current_vertex += 1
            
            for i in range(vertices_bottom):
                if current_vertex >= outer_vertices: break
                x = width/2 - i * (width/(vertices_bottom-1) if vertices_bottom > 1 else 0)
                positions[current_vertex] = (x, -height/2)
                current_vertex += 1
            
            for i in range(vertices_left):
                if current_vertex >= outer_vertices: break
                y = -height/2 + (i+1) * (height/(vertices_left+1))
                positions[current_vertex] = (-width/2, y)
                current_vertex += 1
            
            positions[self.n - 1] = (0, 0)
            
        return positions

    def draw_graph(self, adj_matrix, directed=True):
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph() if directed else nx.Graph()
        
        for i in range(self.n):
            G.add_node(i)
        
        for i in range(self.n):
            for j in range(self.n):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        pos = self.get_vertex_positions()
        
        nx.draw(G, pos, 
               with_labels=True, 
               node_color='lightblue',
               node_size=800,
               arrowsize=20 if directed else 0,
               font_size=12, 
               font_weight='bold',
               arrows=directed,
               edge_color='gray',
               width=1.5,
               node_shape='o')
        
        plt.title(f'{"Directed" if directed else "Undirected"} Graph')
        plt.axis('equal')
        plt.show()

    def print_matrix(self, matrix, title):
        print(f"\n{title}:")
        for row in matrix:
            print(" ".join(map(str, row)))

def main():
    visualizer = GraphVisualizer()
    
    A_dir, A_undir = visualizer.generate_adjacency_matrix()
    
    visualizer.print_matrix(A_dir, "Directed Graph Adjacency Matrix")
    visualizer.print_matrix(A_undir, "Undirected Graph Adjacency Matrix")
    
    visualizer.draw_graph(A_dir, directed=True)
    visualizer.draw_graph(A_undir, directed=False)

if __name__ == "__main__":
    main()