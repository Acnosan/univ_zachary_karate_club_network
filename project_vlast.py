import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class KarateClub:
    def __init__(self, karate_club_nodes):
        self.karate_club_nodes = karate_club_nodes
        self.n_len = len(self.karate_club_nodes)
    # for the adjacency matrix making
    def adjancency_matrix(self) -> np.ndarray :
        adj_mat = np.zeros((self.n_len, self.n_len), dtype=int)
        for node, neighbords in self.karate_club_nodes.items():
            for neighbord in neighbords:
                adj_mat[node-1][neighbord-1] = 1
                adj_mat[neighbord-1][node-1] = 1
        return adj_mat
    
    # to convert the adjacency matrix into networkx graph
    def build_graph(self):
        adj_mat = self.adjancency_matrix()
        return nx.from_numpy_array(adj_mat)
    
    # to visualize the graph
    def visualize(self, graph, node_color="lightblue"):
        #Visualization of the graph
        plt.figure(figsize=(20,12))
        nx.draw(graph, with_labels=True, node_color=node_color, node_size=600, font_size=12)
        plt.title("Zachary's Karate Club Network")
        plt.show()
        
    def graph_metrics(self, graph):
        
        def count_size_order():
            print("="*30)
            adj_mat = self.adjancency_matrix()
            total = 0
            for node in adj_mat:
                total += len(adj_mat[node])
            size = total // 2
            order = len(self.karate_club_nodes)
            
            print(f"Graph Size :{size}, Order : {order}")
            return size,order
        
        def degree_dist(graph):
            print("="*30)
            degree_dist_dict = {}
            for node, degree in graph.degree():
                degree_dist_dict[node] = degree
                print(f"Node {node+1}: {degree} degree")
                
        def clustering_coeff(graph):
            print("="*30)
            clustering_coeff = {}
            c_sum = 0
            print("Average clustering coefficient:", nx.average_clustering(graph))
            for i in range(1,self.n_len):
                neighbors = self.karate_club_nodes[i]
                k = len(neighbors)
                if k<2:
                    return 0
                links = 0
                for u in neighbors:
                    for v in neighbors:
                        if u < v and v in self.karate_club_nodes[u]:
                            links += 1
                clustering_coeff[i] = (2 * links) / (k * (k - 1))
                print(f"Node {i} Coeff : {(clustering_coeff[i]):.2f}")
                c_sum += clustering_coeff[i]
            
            print(f"Manual Average clustering coefficient: {c_sum/self.n_len}")
            return clustering_coeff
        
        def triangle_motif():
            print("="*30)
            adj_mat = self.adjancency_matrix()
            adj_mat_power_3 = np.linalg.matrix_power(adj_mat,3)
            triangle_counter = np.trace(adj_mat_power_3) // 6
            print("Total Number of triangles is :", triangle_counter)
            
            return triangle_counter
        
        def kclique(graph):
            print("="*30)
            cliques = list(nx.find_cliques(graph))
            print("Total Cliques number :", len(cliques))
            
            max_clique = max(cliques, key=len)
            print("Max clique size:", len(max_clique))
            
            print("Max Clique members :")
            for c in cliques:
                if len(c) == max(len(c) for c in cliques):
                    print(c)
            return max_clique
        
        def kcores(graph, k=3):
            core = nx.k_core(graph, k=k)
            print(f"Nodes in {k}-core: {core.nodes()}")
            
            core_numbers = nx.core_number(graph)
            max_core = max(core_numbers.values())
            print(f"Max Value in {k}-core : {max_core}")

            kcore_subgraph = nx.k_core(graph, k=max_core)
            print(f"Nodes of the Max {k}-core : {list(kcore_subgraph.nodes())}")
            return core

        graph_size, graph_order = count_size_order()
        degree_dist_dict = degree_dist(graph)
        clustering_coeff_dict = clustering_coeff(graph)
        triangle_counter = triangle_motif()
        max_kclique = kclique(graph)
        kcores = kcores(graph)
        
    def run_class(self):
        graph = self.build_graph()
        self.visualize(graph)
        self.graph_metrics(graph)