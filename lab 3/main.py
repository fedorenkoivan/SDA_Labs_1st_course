import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Dict, Tuple, Optional

class GraphVisualizer:
    def __init__(self, n3: int = 2, n4: int = 9, seed: Optional[int] = None):
        """
        Initialize GraphVisualizer with configurable parameters.
        
        Args:
            n3 (int): Parameter affecting graph density (default: 2)
            n4 (int): Parameter affecting vertex layout (default: 9)
            seed (int, optional): Random seed for graph generation
        """
        self.validate_parameters(n3, n4)
        self.n3 = n3
        self.n4 = n4
        self.n = 10 + self.n3
        self.k = 1.0 - self.n3 * 0.02 - self.n4 * 0.005 - 0.25
        self.seed = seed

    @staticmethod
    def validate_parameters(n3: int, n4: int) -> None:
        """Validate input parameters."""
        if not isinstance(n3, int) or n3 < 0:
            raise ValueError("n3 must be a non-negative integer")
        if not isinstance(n4, int) or n4 < 0:
            raise ValueError("n4 must be a non-negative integer")

    def generate_adjacency_matrix(self):
        """Generate directed and undirected adjacency matrices."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        T = np.random.uniform(0, 2.0, size=(self.n, self.n))
        A_dir = np.where(T * self.k >= 1.0, 1, 0)
        A_undir = np.maximum(A_dir, A_dir.T)
        
        return A_dir, A_undir

    def get_vertex_positions(self) -> Dict[int, Tuple[float, float]]:
        """Calculate vertex positions for graph visualization."""
        positions = {}
        
        if self.n4 in [8, 9]:
            positions = self._get_rectangular_layout()
        else:
            positions = self._get_circular_layout()
        
        return positions

    def _get_rectangular_layout(self) -> Dict[int, Tuple[float, float]]:
        """Generate rectangular layout for vertices."""
        width, height = 8, 6
        outer_vertices = self.n - 1
        
        vertices_per_side = self._calculate_vertices_per_side(outer_vertices)
        return self._position_vertices_rectangular(vertices_per_side, width, height)


    def _position_vertices_rectangular(self, vertices_per_side: Tuple[int, int, int, int], 
                                    width: float, height: float) -> Dict[int, Tuple[float, float]]:
        """
        Position vertices in a rectangular layout.
        
        Args:
            vertices_per_side: Tuple of (top, right, bottom, left) vertex counts
            width: Width of the rectangle
            height: Height of the rectangle
        """
        positions = {}
        vertices_top, vertices_right, vertices_bottom, vertices_left = vertices_per_side
        current_vertex = 1
        
        for i in range(vertices_top):
            if current_vertex > self.n - 1:
                break
            x = -width/2 + i * (width/(vertices_top-1) if vertices_top > 1 else 0)
            positions[current_vertex] = (x, height/2)
            current_vertex += 1
        
        for i in range(vertices_right):
            if current_vertex > self.n - 1:
                break
            y = height/2 - (i+1) * (height/(vertices_right+1))
            positions[current_vertex] = (width/2, y)
            current_vertex += 1
        
        for i in range(vertices_bottom):
            if current_vertex > self.n - 1:
                break
            x = width/2 - i * (width/(vertices_bottom-1) if vertices_bottom > 1 else 0)
            positions[current_vertex] = (x, -height/2)
            current_vertex += 1
        
        for i in range(vertices_left):
            if current_vertex > self.n - 1:
                break
            y = -height/2 + (i+1) * (height/(vertices_left+1))
            positions[current_vertex] = (-width/2, y)
            current_vertex += 1
        
        positions[self.n] = (0, 0)
        
        return positions

    def _get_circular_layout(self) -> Dict[int, Tuple[float, float]]:
        """Generate circular layout for vertices."""
        positions = {}
        angle_step = 2 * math.pi / self.n
        for i in range(1, self.n + 1):
            angle = i * angle_step
            positions[i] = (math.cos(angle), math.sin(angle))
        return positions

    def _calculate_vertices_per_side(self, outer_vertices: int) -> Tuple[int, int, int, int]:
        """Calculate number of vertices for each side of rectangular layout.""" 
        vertices_top = math.ceil(outer_vertices / 4)
        vertices_right = math.ceil((outer_vertices - vertices_top) / 3)
        vertices_bottom = math.ceil((outer_vertices - vertices_top - vertices_right) / 2)
        vertices_left = outer_vertices - vertices_top - vertices_right - vertices_bottom
        return vertices_top, vertices_right, vertices_bottom, vertices_left

    def draw_graph(self, adj_matrix: np.ndarray, directed: bool = True) -> None:
        """
        Draw the graph using matplotlib.
        
        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph
            directed (bool): Whether to draw a directed graph
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        positions = self.get_vertex_positions()
        for node, (x, y) in positions.items():
            ax.plot(x, y, 'o', markersize=15, color='lightblue', markeredgecolor='black')
            ax.text(x, y, str(node), fontsize=12, ha='center', va='center')

        for i in range(self.n):
            for j in range(self.n):
                if adj_matrix[i][j] == 1:
                    x_start, y_start = positions[i + 1]
                    x_end, y_end = positions[j + 1]

                    # Apply an offset to avoid overlapping with the vertex
                    offset_angle = math.pi / 10  # Angle to avoid overlap
                    x_offset_start = x_start + 0.1 * math.cos(offset_angle)
                    y_offset_start = y_start + 0.1 * math.sin(offset_angle)
                    x_offset_end = x_end + 0.1 * math.cos(offset_angle + math.pi)
                    y_offset_end = y_end + 0.1 * math.sin(offset_angle + math.pi)

                    ax.plot([x_offset_start, x_offset_end], [y_offset_start, y_offset_end], color='gray', lw=1.5)
                    
                    if directed:
                        ax.annotate('', xy=(x_offset_end, y_offset_end), xytext=(x_offset_start, y_offset_start),
                                    arrowprops=dict(arrowstyle="->", lw=1.5))

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(f'{"Directed" if directed else "Undirected"} Graph')
        plt.show()

    @staticmethod
    def print_matrix(matrix: np.ndarray, title: str) -> None:
        """Print adjacency matrix with title."""
        print(f"\n{title}:")
        for row in matrix:
            print(" ".join(map(str, row)))

def main():
    # Example usage with optional seed
    visualizer = GraphVisualizer(n3=2, n4=9, seed=4229)
    
    A_dir, A_undir = visualizer.generate_adjacency_matrix()
    
    visualizer.print_matrix(A_dir, "Directed Graph Adjacency Matrix")
    visualizer.print_matrix(A_undir, "Undirected Graph Adjacency Matrix")
    
    visualizer.draw_graph(A_dir, directed=True)
    visualizer.draw_graph(A_undir, directed=False)

if __name__ == "__main__":
    main()
