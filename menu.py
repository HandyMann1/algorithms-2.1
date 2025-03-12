import tkinter as tk
import random
from tkinter import ttk

import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

global mod


class DirectedGraphPlotter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("The traveling salesman problem - The nearest neighbour heuristic")

        # some key bindings
        self.root.bind("<Control-z>", self.cancel_last_action)
        self.root.bind("<Control-Shift-z>", self.clear_canvas)
        self.root.bind("<Control-Shift-Z>", self.clear_canvas)

        # adding selectable tabs at the top of the window
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Graph creation
        self.graph_frame = tk.Frame(self.notebook)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.graph_frame, text="Graph")

        self.hamilton_frame = tk.Frame(self.notebook)
        self.hamilton_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.hamilton_frame, text="Hamilton Cycle")

        # The table with edges
        self.table_frame = tk.Frame(self.graph_frame)
        self.table_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Lots of initialization
        self.G = nx.DiGraph()
        self.pos = {}
        self.history = []
        self.selected_node = None
        self.selected_edge = None

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.table_frame)
        self.tree['columns'] = ('Source', 'Target', 'Weight')
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Source", anchor=tk.W, width=100)
        self.tree.column("Target", anchor=tk.W, width=100)
        self.tree.column("Weight", anchor=tk.W, width=100)
        self.tree.heading("#0", text='', anchor=tk.W)
        self.tree.heading('Source', text='Source', anchor=tk.W)
        self.tree.heading('Target', text='Target', anchor=tk.W)
        self.tree.heading('Weight', text='Weight', anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.table_frame)
        self.button_frame.pack(fill=tk.X)
        self.select_button = tk.Button(self.button_frame, text="Select Edge", command=self.select_edge)
        self.select_button.pack(side=tk.LEFT)
        self.weight_label = tk.Label(self.button_frame, text="New Weight:")
        self.weight_label.pack(side=tk.LEFT)
        self.weight_entry = tk.Entry(self.button_frame, width=5)
        self.weight_entry.pack(side=tk.LEFT)
        self.update_button = tk.Button(self.button_frame, text="Update Weight", command=self.update_weight)
        self.update_button.pack(side=tk.LEFT)

        self.graph_button_frame = tk.Frame(self.graph_frame)
        self.graph_button_frame.pack(side=tk.BOTTOM)
        self.cancel_button = tk.Button(self.graph_button_frame, text="Cancel Last Action",
                                       command=self.cancel_last_action)
        self.cancel_button.pack(side=tk.LEFT)
        self.clear_button = tk.Button(self.graph_button_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.choose_mod_button = tk.Button(self.graph_button_frame, text="Choose modification", command=self.choose_mod)
        self.choose_mod_button.pack(side=tk.LEFT)

        self.canvas.mpl_connect('button_press_event', self.onclick)

        self.node_select_radius = 0.5

        self.hamilton_button = tk.Button(self.hamilton_frame, text="Find Hamilton Cycle",
                                         command=self.find_hamilton_cycle)
        self.hamilton_button.pack()

        self.hamilton_cycle_text = tk.Text(self.hamilton_frame, height=5, width=50)
        self.hamilton_cycle_text.pack()

        self.hamilton_length_label = tk.Label(self.hamilton_frame, text="Cycle Length: ")
        self.hamilton_length_label.pack()

        self.hamilton_fig = Figure(figsize=(6, 6), dpi=100)
        self.hamilton_ax = self.hamilton_fig.add_subplot(111)
        self.hamilton_canvas = FigureCanvasTkAgg(self.hamilton_fig, master=self.hamilton_frame)
        self.hamilton_canvas.draw()
        self.hamilton_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.edge_weights = {}
        self.mod_var = tk.BooleanVar()  # create BooleanVar

    def onclick(self, event):
        if event.inaxes == self.ax:
            # trying to find the closest node
            closest_node = self.find_closest_node(event.xdata, event.ydata)
            if closest_node:
                if self.selected_node:
                    # Add a directed edge from the selected node to the closest node
                    if (self.selected_node, closest_node) not in self.G.edges():
                        self.G.add_edge(self.selected_node, closest_node)
                        self.history.append((self.selected_node, closest_node))  # Record action
                        # Initialize weight for new edge
                        self.edge_weights[(self.selected_node, closest_node)] = self.calculate_weight(
                            self.selected_node, closest_node)
                    self.selected_node = None
                else:
                    # Select the closest node
                    self.selected_node = closest_node
            else:
                # If no node is selected and none is clicked, add a new node
                if len(self.G.nodes()) < 26:
                    node_name = f"{chr(ord('a') + len(self.G.nodes()))}"
                else:
                    node_name = f"node {len(self.G.nodes()) - 25}"
                self.G.add_node(node_name)
                self.pos[node_name] = (event.xdata, event.ydata)
                self.history.append(node_name)  # Record action
            self.update_plot()

    # we basically try to find the closest node to the place where we clicked,
    # but we keep in mind that this node should be close enough to it, and we should create a new node instead
    def find_closest_node(self, x, y):
        closest_node = None
        min_distance = float('inf')
        for node, pos in self.pos.items():
            distance = ((x - pos[0]) ** 2 + (y - pos[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        return closest_node if min_distance < self.node_select_radius else None

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, 10)
        self.ax.set_ylim(-1, 10)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            ax=self.ax,
            connectionstyle="arc3,rad=0.2",  # we create these good-looking edges
            arrows=True,
            arrowstyle="-|>",
        )
        if self.selected_node:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[self.selected_node], node_color='r', ax=self.ax)
        self.canvas.draw()
        self.update_table()

    # we update the contents of the edge table with this function
    def update_table(self):
        self.tree.delete(*self.tree.get_children())
        for u, v in self.G.edges():
            weight = self.edge_weights.get((u, v), self.calculate_weight(u, v))
            self.tree.insert('', 'end', values=(u, v, weight))

    # to get some weight to the edges when they are created,
    # I use the distance between connected nodes as weight(Idea by FasterXaos)
    def calculate_weight(self, u, v):
        pos_u = self.pos[u]
        pos_v = self.pos[v]
        distance = ((pos_v[0] - pos_u[0]) ** 2 + (pos_v[1] - pos_u[1]) ** 2) ** 0.5
        return round(distance, 2)

    def cancel_last_action(self, event=None):
        if self.history:
            last_action = self.history.pop()
            if isinstance(last_action, tuple):  # edges in self.history are stored as tuples so this thing removes edges
                if last_action in self.G.edges():
                    self.G.remove_edge(last_action[0], last_action[1])
                if (last_action[0], last_action[1]) in self.edge_weights:
                    del self.edge_weights[(last_action[0], last_action[1])]
            else:  # except edges, we store nodes, so this one removes them
                if last_action in self.G.nodes():
                    self.G.remove_node(last_action)
                del self.pos[last_action]
                if last_action == self.selected_node:
                    self.selected_node = None
            self.update_plot()

    # just  getting rid of the mess we made on the canvas via button
    def clear_canvas(self,event=None):
        self.G.clear()
        self.pos.clear()
        self.history.clear()
        self.selected_node = None
        self.edge_weights.clear()
        self.update_plot()

    def choose_mod(self):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Choose modification type:")

        modification_label = tk.Label(selection_window, text="Choose modification type:")
        modification_label.pack(pady=10)

        no_modification_radio = ttk.Radiobutton(selection_window, text="Classic nearest neighbor algorithm",
                                                variable=self.mod_var,
                                                value=False)
        modification_radio = ttk.Radiobutton(selection_window, text="Repetitive nearest neighbor algorithm",
                                             variable=self.mod_var,
                                             value=True)
        no_modification_radio.pack(pady=5)
        modification_radio.pack(pady=5)

        confirm_button = ttk.Button(selection_window, text="Confirm",
                                    command=lambda: self.confirm_mod(selection_window))
        confirm_button.pack(pady=20)

    def confirm_mod(self, selection_window):
        global mod
        mod = self.mod_var.get()
        selection_window.destroy()  # close window

    # selectin' the edge to change its weight
    def select_edge(self):
        selected_item = self.tree.focus()
        if selected_item:
            values = self.tree.item(selected_item, 'values')
            if values:
                self.selected_edge = (values[0], values[1])

    # and after typing in our edge weight in the entry we need to update it by pressing that button
    def update_weight(self):
        if self.selected_edge:
            try:
                new_weight = float(self.weight_entry.get())
                self.edge_weights[self.selected_edge] = new_weight  # we update that weight in the weight dict
                self.update_table()  # and we update the edge table
            except ValueError:
                print("Invalid weight entered.")
            self.selected_edge = None

    # finally, here lies the logic of nearest neighbour heuristic to find the shortest Hamilton cycle
    def find_hamilton_cycle(self):
        if not self.G.nodes():  # just checkin' for empty cycle
            self.hamilton_cycle_text.delete("1.0", tk.END)
            self.hamilton_cycle_text.insert(tk.END, "Graph is empty!")
            self.hamilton_length_label.config(text="Cycle Length: N/A")
            self.update_hamilton_plot(None)
            return

        # Create a cost matrix for the graph
        cost_matrix = {}
        for u, v in self.G.edges():
            weight = self.edge_weights.get((u, v), self.calculate_weight(u, v))
            cost_matrix[(u, v)] = weight

        def find_hamiltonian_cycle(graph, mod=True):
            def find_path(current_path, current_cost):
                if len(current_path) == len(graph.nodes()):  # this is how we end the path finding
                    first_node = current_path[0]
                    last_node = current_path[-1]
                    if graph.has_edge(last_node, first_node):
                        edge = (last_node, first_node)
                        cost = cost_matrix.get(edge, self.calculate_weight(last_node, first_node))
                        return current_path + [first_node], current_cost + cost
                    else:
                        return None, float('inf')

                last_node = current_path[-1]
                best_path = None
                min_cost = float('inf')
                for neighbor in graph.neighbors(last_node):
                    if neighbor not in current_path:
                        edge = (last_node, neighbor)
                        cost = cost_matrix.get(edge, self.calculate_weight(last_node, neighbor))
                        path, total_cost = find_path(current_path + [neighbor], current_cost + cost)
                        if path and total_cost < min_cost:
                            min_cost = total_cost
                            best_path = path

                return best_path, min_cost

            best_cycle = None
            min_cycle_cost = float('inf')
            if mod is True:
                for start_node in graph.nodes():
                    cycle, cycle_cost = find_path([start_node], 0)
                    if cycle and cycle_cost < min_cycle_cost:
                        min_cycle_cost = cycle_cost
                        best_cycle = cycle
            else:
                start_node = random.choice(list(graph.nodes()))
                best_cycle, min_cycle_cost = find_path([start_node], 0)

            return best_cycle, min_cycle_cost

        # find the optimal Hamilton cycle
        hamilton_cycle, total_length = find_hamiltonian_cycle(self.G, mod=mod)

        if hamilton_cycle:
            # display the result
            cycle_str = " -> ".join(hamilton_cycle)
            self.hamilton_cycle_text.delete("1.0", tk.END)
            self.hamilton_cycle_text.insert(tk.END, cycle_str)
            self.hamilton_length_label.config(text=f"Cycle Length: {total_length:.2f}")  # update length label
            self.update_hamilton_plot(hamilton_cycle)
        else:
            self.hamilton_cycle_text.delete("1.0", tk.END)
            self.hamilton_cycle_text.insert(tk.END, "No Hamilton cycle found.")
            self.hamilton_length_label.config(text="Cycle Length: N/A")
            self.update_hamilton_plot(None)

    def update_hamilton_plot(self, path):
        self.hamilton_ax.clear()
        self.hamilton_ax.set_xlim(-1, 10)
        self.hamilton_ax.set_ylim(-1, 10)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.hamilton_ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.hamilton_ax)

        if path:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=edges,
                ax=self.hamilton_ax,
                edge_color='r',
                width=2,
                connectionstyle="arc3,rad=0.2",
                arrows=True,
                arrowstyle="-|>"
            )
        self.hamilton_canvas.draw()

    def run(self):
        self.root.mainloop()


mod = False
plotter = DirectedGraphPlotter()
plotter.run()
