import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class KarateClub:
    def __init__(self, karate_club_nodes):
        self.karate_club_nodes = karate_club_nodes
        self.n_len = len(self.karate_club_nodes)

    def adjacency(self) -> np.ndarray :
        """ making the adjacency matrix """
        adj_mat = np.zeros((self.n_len, self.n_len), dtype=int)
        for node, neighbords in self.karate_club_nodes.items():
            for neighbord in neighbords:
                adj_mat[node-1][neighbord-1] = 1
                adj_mat[neighbord-1][node-1] = 1
        return adj_mat
    
    
    def build_graph(self):
        """ to convert the adjacency matrix into networkx graph """
        adj_mat = self.adjacency()
        return nx.from_numpy_array(adj_mat)
    
    
    def visualize(self, graph, node_color="lightblue"):
        
        """ Visualization of the graph """
        plt.figure(figsize=(18,10))
        nx.draw(graph, with_labels=True, node_color=node_color, node_size=600, font_size=12)
        plt.title("Zachary's Karate Club Network")
        plt.show()
        
    def graph_metrics(self, graph):
        
        def count_size_order(graph):
            """ counting of the size and order of the graph"""
            print("="*30)
            size = graph.number_of_edges()
            order = len(self.karate_club_nodes)
            print(f"Graph Size :{size}, Order : {order}")
            return size,order
        
        def degree_dist(graph):
            """ counting the degree distribution for each node"""
            print("="*30)
            degree_dist_dict = {}
            for node, degree in graph.degree():
                degree_dist_dict[node] = degree
                print(f"Node {node+1}: {degree} degree")
                
        def clustering_coeff(graph):
            """ counting the clustering coefficient for each node and the average """
            print("="*30)
            clustering_coeff = {}
            c_sum = 0
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
            print("Nx Average clustering coefficient:", nx.average_clustering(graph))
            print(f"Manual Average clustering coefficient: {c_sum/self.n_len}")
            return clustering_coeff
        
        def triangle_motif():
            """ counting the motif of the traingles """
            print("="*30)
            adj_mat = self.adjacency()
            adj_mat_power_3 = np.linalg.matrix_power(adj_mat,3)
            triangle_counter = np.trace(adj_mat_power_3) // 6
            print("Total Number of triangles is :", triangle_counter)
            
            return triangle_counter
        
        def kclique(graph):
            """ counting the k cliques """
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
            """ counting the k cores """
            core = nx.k_core(graph, k=k)
            print(f"Nodes in {k}-core: {core.nodes()}")
            
            core_numbers = nx.core_number(graph)
            max_core = max(core_numbers.values())
            print(f"Max Value in {k}-core : {max_core}")

            kcore_subgraph = nx.k_core(graph, k=max_core)
            print(f"Nodes of the Max {k}-core : {list(kcore_subgraph.nodes())}")
            return core

        graph_size, graph_order = count_size_order(graph)
        degree_dist_dict = degree_dist(graph)
        clustering_coeff_dict = clustering_coeff(graph)
        triangle_counter = triangle_motif()
        max_kclique = kclique(graph)
        kcores = kcores(graph)
        
    def run_class(self):
        graph = self.build_graph()
        self.visualize(graph)
        self.graph_metrics(graph)
if __name__ == "__main__":
    karate_club_nodes = {
    1: [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32],
    2: [1, 3, 4, 8, 14, 18, 20, 22, 31],
    3: [1, 2, 4, 8, 9, 10, 14, 28, 29, 33],
    4: [1, 2, 3, 8, 13, 14],
    5: [1, 7, 11],
    6: [1, 7, 11, 17],
    7: [1, 5, 6, 17],
    8: [1, 2, 3, 4],
    9: [1, 3, 31, 33, 34],
    10: [3, 34],
    11: [1, 5, 6],
    12: [1],
    13: [1, 4],
    14: [1, 2, 3, 4, 34],
    15: [33, 34],
    16: [33, 34],
    17: [6, 7],
    18: [1, 2],
    19: [33, 34],
    20: [1, 2, 34],
    21: [33, 34],
    22: [1, 2],
    23: [33, 34],
    24: [26, 28, 30, 33, 34],
    25: [26, 28, 32],
    26: [24, 25, 32],
    27: [30, 34],
    28: [3, 24, 25],
    29: [3, 32, 33],
    30: [24, 27, 33, 34],
    31: [2, 9, 33, 34],
    32: [1, 25, 26, 29, 33, 34],
    33: [3, 9, 15, 16, 19, 21, 23, 24, 29, 30, 31, 32, 34],
    34: [9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 30, 31, 32, 33]
    }
    myclass = KarateClub(karate_club_nodes)
    myclass.run_class()