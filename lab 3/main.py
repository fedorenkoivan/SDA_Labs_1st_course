import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Rectangle
import math

class GraphVisualizer:
    def __init__(self, variant):
        # Parse variant number
        self.n1 = variant // 1000
        self.n2 = (variant // 100) % 10
        self.n3 = (variant // 10) % 10
        self.n4 = variant % 10
        
        # Calculate number of vertices
        self.n = 10 + self.n3
        
        # Calculate coefficient k
        self.k = 1.0 - self.n3 * 0.02 - self.n4 * 0.005 - 0.25

    def generate_adjacency_matrix(self):
        # Set random seed
        np.random.seed(self.n1 * 1000 + self.n2 * 100 + self.n3 * 10 + self.n4)
        
        # Generate random matrix
        T = np.random.uniform(0, 2.0, size=(self.n, self.n))
        
        # Apply coefficient and round
        A_dir = np.where(T * self.k >= 1.0, 1, 0)
        
        # Create undirected matrix
        A_undir = np.maximum(A_dir, A_dir.T)
        
        return A_dir, A_undir

    def get_vertex_positions(self):
        positions = {}
        if self.n4 in [8, 9]:  # Rectangle with center vertex
            width, height = 8, 6
            vertices_per_side = math.ceil((self.n - 1) / 4)
            
            # Calculate positions for vertices on the rectangle
            current_vertex = 0
            # Top side
            for i in range(vertices_per_side):
                if current_vertex >= self.n - 1: break
                x = -width/2 + i * (width/(vertices_per_side-1))
                positions[current_vertex] = (x, height/2)
                current_vertex += 1
            
            # Right side
            for i in range(vertices_per_side):
                if current_vertex >= self.n - 1: break
                y = height/2 - i * (height/(vertices_per_side-1))
                positions[current_vertex] = (width/2, y)
                current_vertex += 1
            
            # Bottom side
            for i in range(vertices_per_side):
                if current_vertex >= self.n - 1: break
                x = width/2 - i * (width/(vertices_per_side-1))
                positions[current_vertex] = (x, -height/2)
                current_vertex += 1
            
            # Left side
            for i in range(vertices_per_side):
                if current_vertex >= self.n - 1: break
                y = -height/2 + i * (height/(vertices_per_side-1))
                positions[current_vertex] = (-width/2, y)
                current_vertex += 1
            
            # Center vertex
            positions[self.n - 1] = (0, 0)
        
        return positions

    def draw_graph(self, adj_matrix, directed=True):
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add nodes
        for i in range(self.n):
            G.add_node(i)
        
        # Add edges
        for i in range(self.n):
            for j in range(self.n):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        # Get positions
        pos = self.get_vertex_positions()
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, arrowsize=20 if directed else 0,
                font_size=12, font_weight='bold')
        
        plt.title(f'{"Directed" if directed else "Undirected"} Graph')
        plt.axis('equal')
        plt.show()

    def print_matrix(self, matrix, title):
        print(f"\n{title}:")
        for row in matrix:
            print(" ".join(map(str, row)))

def main():
    # Initialize with your variant
    variant = 4229
    visualizer = GraphVisualizer(variant)
    
    # Generate matrices
    A_dir, A_undir = visualizer.generate_adjacency_matrix()
    
    # Print matrices
    visualizer.print_matrix(A_dir, "Directed Graph Adjacency Matrix")
    visualizer.print_matrix(A_undir, "Undirected Graph Adjacency Matrix")
    
    # Draw graphs
    visualizer.draw_graph(A_dir, directed=True)
    visualizer.draw_graph(A_undir, directed=False)

if __name__ == "__main__":
    main()