import networkx as nx
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import scrolledtext

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from typing import Optional
import csv
import os

from project_vlast import KarateClub

class TkinterInterface:
    
    def __init__(self, G, metrics):
        self.G = G
        self.metrics = metrics
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.entry_node: Optional[tk.Entry] = None
        self.entry_edge: Optional[tk.Entry] = None
        self.fig: Optional[Figure] = None
        self.start_interface()
        
    def draw_graph(self):
        """Draw the graph on the canvas."""
        if self.canvas is None or self.fig is None:
            return
            
        # Clear the figure, not just the plot
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color="skyblue", 
                node_size=600, font_size=10, ax=ax)
        self.canvas.draw()

    def add_node(self):
        """Add a new node to the graph."""
        if self.entry_node is None:
            return
            
        try:
            node = int(self.entry_node.get())
            if node in self.G:
                messagebox.showwarning("Error", f"Node {node} already exists.")
            else:
                self.G.add_node(node)
                messagebox.showinfo("Success", f"Node {node} added.")
                self.draw_graph()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid node number.")

    def remove_node(self):
        """Remove a node from the graph."""
        if self.entry_node is None:
            return
            
        try:
            node = int(self.entry_node.get())
            if node in self.G:
                self.G.remove_node(node)
                messagebox.showinfo("Success", f"Node {node} removed.")
                self.draw_graph()
            else:
                messagebox.showwarning("Error", f"Node {node} does not exist.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid node number.")

    def add_edge(self):
        """Add a new edge between two nodes."""
        if self.entry_edge is None:
            return
            
        try:
            n1, n2 = map(int, self.entry_edge.get().split())
            if self.G.has_edge(n1, n2):
                messagebox.showwarning("Error", "This edge already exists.")
            else:
                self.G.add_edge(n1, n2)
                messagebox.showinfo("Success", f"Edge ({n1}, {n2}) added.")
                self.draw_graph()
        except:
            messagebox.showerror("Error", "Enter two nodes separated by a space.")

    def remove_edge(self):
        """Remove an edge between two nodes."""
        if self.entry_edge is None:
            return
            
        try:
            n1, n2 = map(int, self.entry_edge.get().split())
            if self.G.has_edge(n1, n2):
                self.G.remove_edge(n1, n2)
                messagebox.showinfo("Success", f"Edge ({n1}, {n2}) removed.")
                self.draw_graph()
            else:
                messagebox.showwarning("Error", "This edge does not exist.")
        except:
            messagebox.showerror("Error", "Enter two nodes separated by a space.")
    
    def show_info(self):
        """Display graph statistics """
        def flat_dict():
            flat_metrics = {}
            for key, val in self.metrics.items():
                if isinstance(val, dict) and all(isinstance(v, dict) for v in val.values()):
                    for subkey, subval in val.items():
                        flat_metrics[subkey] = subval
                else:
                    flat_metrics[key] = val
            return flat_metrics
        
        def fit_in_window(title, text):
            """Show resizable scrollable window instead of messagebox."""
            win = tk.Toplevel()
            win.title(title)
            win.geometry("1280x720")  
            win.resizable(True, True)

            # Scrollable text area
            txt = scrolledtext.ScrolledText(win, wrap="none", font=("Courier New", 10))
            txt.insert("1.0", text)
            txt.configure(state="disabled")  # make read-only
            txt.pack(expand=True, fill="both")

            # Optional close button
            btn = tk.Button(win, text="Close", command=win.destroy)
            btn.pack(pady=5)
            
        results = flat_dict()
        lines = []

        # Simple scalar metrics
        simple = [f"{k}: {v}" for k, v in results.items() if not isinstance(v, (dict, list))]
        if simple:
            lines.append("\n".join(simple))
            lines.append("")

        # extracting metric dictionaries
        dicts = {k: v for k, v in results.items() if isinstance(v, dict)}
        if dicts:
            col_width = 30

            # sort by node
            sorted_dicts = {k: dict(sorted(v.items())) for k, v in dicts.items()}
            max_rows = max(len(d) for d in sorted_dicts.values())

            # headers (first column = node ID)
            headers = ["Node".ljust(col_width)] + [f"{k:<{col_width}}" for k in sorted_dicts.keys()]
            lines.append("".join(headers))
            lines.append("-" * (col_width * (len(sorted_dicts))))

            # Build each row
            for i in range(max_rows):
                node_label = f"node {i+1}".ljust(col_width)
                row = [node_label]
                for _, dict_vals in sorted_dicts.items():
                    items = list(dict_vals.items())
                    # if we still have rows to write into
                    if i < len(items):
                        _, value = items[i]
                        if isinstance(value, (int, float)):
                            cell = f"{value:.2f}" 
                        else:
                            cell = f"{value}" 
                    else:
                        cell = ""
                    row.append(f"{cell:<{col_width}}")
                lines.append("".join(row))

        # Lists at end
        lists = [f"{k}: {v}" for k, v in results.items() if isinstance(v, list)]
        if lists:
            lines.append("")
            lines.extend(lists)
        
        fit_in_window("Graph Information", "\n".join(lines))

    def export_to_csv(self):
        """Export graph data to CSV file."""
        try:
            # Ask user where to save
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Graph Data as CSV"
            )
            
            if not file_path:
                return  # User cancelled
            
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # for nodes and centrality mesures
                writer.writerow([
                    "Node",
                    "Degree Dist",
                    "Clustering Coefficient",
                    "Cent Degree", "Cent Closeness", "Centy Betweenness", "Cent Eigenvector"
                ])
                
                for node in self.G.nodes():
                    writer.writerow([
                        node,
                        self.metrics['Degree Dist'][node],
                        self.metrics['Clustering Coeff'][node],
                        self.metrics["Centrality Mesures"]["Cent Degree"][node],
                        self.metrics["Centrality Mesures"]["Cent Closeness"][node],
                        self.metrics["Centrality Mesures"]["Cent Betweenness"][node],
                        self.metrics["Centrality Mesures"]["Cent Eigenvector"][node],
                        ])
                writer.writerow([])
                
                # for edges
                writer.writerow(["Source", "Target"])
                for edge in self.G.edges():
                    writer.writerow([edge[0], edge[1]])
                
                writer.writerow([])
                
                # for statistics
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Nodes", self.metrics["Order (nodes)"]])
                writer.writerow(["Total Edges", self.metrics["Size (edges)"]])
                writer.writerow(["Average Clustering", self.metrics["Avg Clustering"]])
                writer.writerow(["Density", f"{nx.density(self.G):.3f}"])
                writer.writerow(["Total Triangles", self.metrics["Triangles Count"]])
                writer.writerow(["Max Kclique", self.metrics["Max Kclique"]])
                
                # for max / most centrality mesures
                writer.writerow(["Most central (degree)", self.metrics["Max Cent Degree"]])
                writer.writerow(["Most central (closeness)", self.metrics["Max Cent Closeness"]])
                writer.writerow(["Most central (betweenness)", self.metrics["Max Cent Betweenness"]])
                writer.writerow(["Most central (eigenvector)", self.metrics["Max Cent Eigenvector"]])

            messagebox.showinfo("Success", f"Data exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
    
    def export_to_image(self):
        """Export graph visualization to image file."""
        try:
            # Ask user where to save
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ],
                title="Save Graph Visualization"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Create a new figure for export (higher quality)
            export_fig = plt.figure(figsize=(16, 12))
            ax = export_fig.add_subplot(111)
            
            pos = nx.spring_layout(self.G)
            nx.draw(self.G, pos, with_labels=True, node_color="skyblue", 
                    node_size=800, font_size=10, font_weight='bold', ax=ax)
            
            # Add title
            ax.set_title("Karate Club Network", fontsize=16, fontweight='bold')
            
            # Save with high DPI
            export_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(export_fig)
            
            messagebox.showinfo("Success", f"Visualization exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export image:\n{str(e)}")
        
    def start_interface(self):
        """Initialize and start the GUI."""
        root = tk.Tk()
        root.title("Network Analysis - Karate Club")

        self.fig = plt.figure(figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # Input frame
        frame = tk.Frame(root)
        frame.pack()

        # Node management
        tk.Label(frame, text="Node:").grid(row=0, column=0)
        self.entry_node = tk.Entry(frame, width=10)
        self.entry_node.grid(row=0, column=1)

        tk.Button(frame, text="Add Node", 
                command=self.add_node).grid(row=0, column=2)
        tk.Button(frame, text="Remove Node", 
                command=self.remove_node).grid(row=0, column=3)

        # Edge management
        tk.Label(frame, text="Edge (n1 n2):").grid(row=1, column=0)
        self.entry_edge = tk.Entry(frame, width=10)
        self.entry_edge.grid(row=1, column=1)

        tk.Button(frame, text="Add Edge", 
                command=self.add_edge).grid(row=1, column=2)
        tk.Button(frame, text="Remove Edge", 
                command=self.remove_edge).grid(row=1, column=3)
        
        # Info and Export buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        # Info button
        tk.Button(button_frame, text="Show Info All", 
                command=self.show_info).pack(padx=5)
        tk.Button(button_frame, text="Export to CSV", 
                        command=self.export_to_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Export Graph to Image", 
                command=self.export_to_image).pack(side=tk.LEFT, padx=5)

        self.draw_graph()
        root.mainloop()

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
    graph_class = KarateClub(karate_club_nodes)
    graph = graph_class.build_graph()
    metrics = graph_class.graph_metrics(graph)
    interface = TkinterInterface(graph, metrics)