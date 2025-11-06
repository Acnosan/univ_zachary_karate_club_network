import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List, Any

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
    
    def build_graph(self, G=None):
        """ Build or update graph """
        if G is None:
            # Initial build from adjacency matrix
            adj_mat = self.adjacency()
            mapping = {i: i+1 for i in range(self.n_len)}
            graph = nx.relabel_nodes(nx.from_numpy_array(adj_mat), mapping)
        else:
            # Use existing modified graph
            graph = G
        self.n_len = graph.number_of_nodes()
        return graph
    
    def visualize(self, graph, node_color="lightblue"):
        """ Visualization of the graph """
        plt.figure(figsize=(12,4))
        nx.draw(graph, with_labels=True, node_color=node_color, node_size=600, font_size=12)
        plt.title("Zachary's Karate Club Network")
        plt.show()
    
    def graph_metrics(self, graph):
        
        def count_size_order(graph) -> Tuple[int, int]:
            """ counting of the size and order of the graph"""
            print("SIZE AND ORDER")
            print("="*30)
            order = int(self.n_len)
            size = int(graph.number_of_edges())
            
            print(f"Graph Size :{size}, Order : {order}")
            return order, size
        
        def degree_dist(graph) -> Dict[int, int]:
            """ counting the degree distribution for each node"""
            print("="*30)
            print("DEGREES DISTRIBUTION")
            degree_dist_dict = {}
            for node, degree in graph.degree():
                degree_dist_dict[node] = int(degree)
                print(f"Node {node+1} degree: {degree}")
            return degree_dist_dict
        
        def clustering_coeff(graph) -> Tuple[Dict[int, float], float]:
            """ counting the clustering coefficient for each node and the average """
            print("="*30)
            print("CLUSTERINGS")
            clustering_coeff_dict = {}
            c_sum = 0
            for i in range(1,self.n_len+1):
                neighbors = graph[i]
                k = len(neighbors)
                # for nodes with less than 2 neighbors
                if k<2:
                    clustering_coeff_dict[i] = 0.0
                    print(f"Node {i} Coeff : {clustering_coeff_dict[i]:.2f}")
                    c_sum += clustering_coeff_dict[i]
                    continue 
                links = 0
                for u in neighbors:
                    for v in neighbors:
                        if u < v and v in graph[u]:
                            links += 1
                clustering_coeff_dict[i] = round((2 * links) / (k * (k - 1)), 2)
                print(f"Node {i} Coeff : {(clustering_coeff_dict[i]):.2f}")
                c_sum += clustering_coeff_dict[i]
            average_clustering = round(c_sum/self.n_len,2)
            print(f"Nx Average clustering coefficient: {round(nx.average_clustering(graph),2)}")
            print(f"Manual Average clustering coefficient: {average_clustering}")
            return clustering_coeff_dict, average_clustering
        
        def triangle_motif() -> int:
            """ counting the motif of the traingles """
            print("="*30)
            print("MOTIFS (TRIANGLE)")
            adj_mat = self.adjacency()
            adj_mat_power_3 = np.linalg.matrix_power(adj_mat,3)
            triangles_counter = np.trace(adj_mat_power_3) // 6
            print("Total Number of triangles is :", triangles_counter)
            
            return triangles_counter
        
        def kclique(graph) -> List[int]:
            """ counting the k cliques """
            print("="*30)
            print("K CLIQUES")
            cliques = list(nx.find_cliques(graph))
            print("Total Cliques number :", len(cliques))
            
            max_clique = max(cliques, key=len)
            print(f"Max Clique {max_clique}")
            print(f"Max Clique size: {len(max_clique)}")

            return max_clique
        
        def kcores(graph, k=3):
            """ counting the k cores """
            print("="*30)
            print("K CORES")
            core = nx.k_core(graph, k=k)
            print(f"Nodes in {k}-core: {[core for core in core.nodes()]}")
            
            core_numbers = nx.core_number(graph)
            max_core = max(core_numbers.values())
            print(f"Max Value in {k}-core : {max_core}")
            
            for node, core in core_numbers.items():
                print(f"Node {node} core : {core}")

            kcore_subgraph = nx.k_core(graph, k=max_core)
            print(f"Nodes of the Max {k}-core : {[ node for node in kcore_subgraph.nodes()]}")
            return core

        def centrality_mesures(graph) -> Tuple[Dict[str,Dict[int, Any]], Dict[str, str]]:
            """ counting the centrality mesures"""
            print("="*30)
            print("CENTRALITY")

            # Degree Centrality
            degree_centrality = nx.degree_centrality(graph)
            max_node_deg = int(max(degree_centrality, key=lambda x: degree_centrality[x]))
            print(f"Most central (degree): Node {max_node_deg} : {degree_centrality[max_node_deg]:.2f}")
            
            # Closeness Centrality
            closeness_centrality = nx.closeness_centrality(graph)
            max_node_close = int(max(closeness_centrality, key=lambda x: closeness_centrality[x]))
            print(f"Most central (closeness): Node {max_node_close} :{closeness_centrality[max_node_close]:.2f}")
            
            # Betweenness Centrality
            betweenness_centrality = nx.betweenness_centrality(graph)
            max_node_bet = int(max(betweenness_centrality, key=lambda x: betweenness_centrality[x]))
            print(f"Most central (betweenness): Node {max_node_bet} : {betweenness_centrality[max_node_bet]:.2f}")

            # Eigenvector Centrality
            eigenvector_centrality = nx.eigenvector_centrality(graph)
            max_node_eig = int(max(eigenvector_centrality, key=lambda x: eigenvector_centrality[x]))
            print(f"Most central (eigenvector): Node {max_node_eig} : {eigenvector_centrality[max_node_eig]:.2f}")

            return (
                {
                "Cent Degree": degree_centrality,
                "Cent Closeness": closeness_centrality,
                "Cent Betweenness": betweenness_centrality,
                "Cent Eigenvector": eigenvector_centrality,
                },
                {
                "Max Cent Degree": f"Node {max_node_deg} : {degree_centrality[max_node_deg]:.2f}", 
                "Max Cent Closeness":  f"Node {max_node_close} : {closeness_centrality[max_node_close]:.2f}", 
                "Max Cent Betweenness":  f"Node {max_node_bet} : {betweenness_centrality[max_node_bet]:.2f}", 
                "Max Cent Eigenvector":  f"Node {max_node_eig} : {eigenvector_centrality[max_node_eig]:.2f}", 
                }
            )
        
        graph_order, graph_size = count_size_order(graph)
        degree_dist_dict = degree_dist(graph)
        clustering_coeff_dict, average_clustering = clustering_coeff(graph)
        triangles_counter = triangle_motif()
        max_kclique = kclique(graph)
        kcores = kcores(graph)
        cent_mesures_dict, max_cent_mesures = centrality_mesures(graph)
        
        results = {
            "Order (nodes)" : graph_order,
            "Size (edges)" : graph_size,
            "Degree Dist" : degree_dist_dict,
            "Clustering Coeff" : clustering_coeff_dict,
            "Avg Clustering" : average_clustering,
            "Triangles Count" : triangles_counter,
            "Max Kclique" : max_kclique,
            "Centrality Mesures": cent_mesures_dict,
            "Max Cent Degree": max_cent_mesures["Max Cent Degree"],
            "Max Cent Closeness": max_cent_mesures["Max Cent Closeness"],
            "Max Cent Betweenness": max_cent_mesures["Max Cent Betweenness"],
            "Max Cent Eigenvector": max_cent_mesures["Max Cent Eigenvector"],
        }
        return results
    
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