## Course Labs Overview

This repository contains six progressively complex laboratory assignments, each focusing on a core area of algorithmic and data-structure design, with an emphasis on C implementation and visualization.

---

### Lab 1: Recursive Series Computation
- **Objective:** Compute square roots using Taylor (Maclaurin) series expansions.  
- **Tasks:**
  - Implement **descending recursion**.
  - Implement **ascending recursion**.
  - Implement a **hybrid** approach combining both.
  - Compare all recursive versions with an **iterative** method.  
- **Key Concepts:** recursion strategies, numerical approximation, convergence behavior.

---

### Lab 2: Linked List Manipulation
- **Objective:** Master dynamic singly linked lists in C.  
- **Tasks:**
  - Create and initialize a linked list from user input.
  - Traverse and print list elements.
  - Implement a **block-rearrangement** algorithm that reorders elements in configurable patterns.  
- **Key Concepts:** pointers, memory management (`malloc`/`free`), list operations.

---

### Lab 3: Graph Representation
- **Objective:** Visualize directed and undirected graphs using adjacency matrices.  
- **Tasks:**
  - Generate an **adjacency matrix** for a given vertex count and edge set.
  - Position vertices on a rectangle’s perimeter with one central vertex.
  - Render edges as arrowed (directed) or plain lines (undirected).  
- **Key Concepts:** matrix representation, coordinate placement, basic graphics (console or GUI).

---

### Lab 4: Graph Analysis
- **Objective:** Perform advanced analysis on directed graphs.  
- **Tasks:**
  - **Regular graph detection** (all vertices same degree).
  - Identify **hanging** (degree = 1) and **isolated** (degree = 0) vertices.
  - Find **all simple paths** of specified lengths.
  - Compute **reachability** and **strong connectivity** matrices.
  - Detect **strongly connected components** and build the **condensation graph**.  
- **Key Concepts:** matrix exponentiation, graph traversal, Kosaraju’s or Tarjan’s algorithm.

---

### Lab 5: Graph Traversal Algorithms
- **Objective:** Implement and visualize BFS and DFS.  
- **Tasks:**
  - Write **Breadth-First Search** with a queue.
  - Write **Depth-First Search** (recursive or stack-based).
  - Provide **step-by-step** visual feedback on vertex visitation order.  
- **Key Concepts:** traversal order, use of queue vs. recursion/stack, animation of steps.

---

### Lab 6: Minimum Spanning Tree
- **Objective:** Find and display a Minimum Spanning Tree (MST) of a weighted graph.  
- **Tasks:**
  - Implement **Prim’s algorithm** using a priority queue or simple array.
  - Visualize each step as edges are added to the MST.
  - (Optional) Compare with **Kruskal’s algorithm**.  
- **Key Concepts:** greedy algorithms, priority queues, edge-weight comparison, interactive visualization.

---

> **Note:** Each lab builds on the previous ones—starting from fundamental recursion techniques (Lab 1), moving through linked lists (Lab 2), and culminating in sophisticated graph algorithms and visualizations (Labs 3–6).

---

## General Usage

### Compiling and Running C Programs
1. **Compile** your `.c` source file:
   ```bash
   gcc -Wall -Wextra -std=c11 -o labX labX.c
