import math
import tkinter as tk
import random
from tkinter import ttk

import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

global mod_nn
global mod_ann


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

        # graph creation
        self.graph_frame = tk.Frame(self.notebook)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.graph_frame, text="Graph")

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

        self.canvas.mpl_connect('button_press_event', self.onclick)

        self.node_select_radius = 0.5

        # hamilton nearest neighbour
        self.nn_frame = tk.Frame(self.notebook)
        self.nn_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.nn_frame, text="nearest neighbor")

        self.nn_hamilton_button = tk.Button(self.nn_frame, text="Find Hamilton Cycle",
                                            command=self.nn_find_hamilton_cycle)
        self.nn_hamilton_button.pack()

        self.nn_hamilton_cycle_text = tk.Text(self.nn_frame, height=5, width=50)
        self.nn_hamilton_cycle_text.pack()

        nn_modification_label = tk.Label(self.nn_frame, text="Choose modification type:")
        nn_modification_label.pack(pady=10)

        global mod_nn
        nn_no_modification_radio = ttk.Radiobutton(self.nn_frame, text="Classic nearest neighbor algorithm",
                                                   variable=mod_nn,
                                                   value=False)
        nn_modification_radio = ttk.Radiobutton(self.nn_frame, text="Repetitive nearest neighbor algorithm",
                                                variable=mod_nn,
                                                value=True)
        nn_no_modification_radio.pack(pady=5)
        nn_modification_radio.pack(pady=5)

        nn_confirm_button = ttk.Button(self.nn_frame, text="Confirm", command=self.nn_confirm_mod)
        nn_confirm_button.pack(pady=10)

        self.nn_hamilton_length_label = tk.Label(self.nn_frame, text="Cycle Length: ")
        self.nn_hamilton_length_label.pack()

        self.nn_hamilton_fig = Figure(figsize=(6, 6), dpi=100)
        self.nn_hamilton_ax = self.nn_hamilton_fig.add_subplot(111)
        self.nn_hamilton_canvas = FigureCanvasTkAgg(self.nn_hamilton_fig, master=self.nn_frame)
        self.nn_hamilton_canvas.draw()
        self.nn_hamilton_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.edge_weights = {}
        self.nn_mod_var = tk.BooleanVar()

        # hamilton simulated annealing
        self.ann_frame = tk.Frame(self.notebook)
        self.ann_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.ann_frame, text="simulated annealing")

        self.ann_hamilton_button = tk.Button(self.ann_frame, text="Find Hamilton Cycle",
                                             command=self.ann_find_hamilton_cycle)
        self.ann_hamilton_button.pack()

        self.ann_hamilton_cycle_text = tk.Text(self.ann_frame, height=5, width=50)
        self.ann_hamilton_cycle_text.pack()

        ann_modification_label = tk.Label(self.ann_frame, text="Choose modification type:")
        ann_modification_label.pack(pady=10)

        global mod_ann
        ann_no_modification_radio = ttk.Radiobutton(self.ann_frame, text="Classic simulated annealing algorithm",
                                                    variable=mod_ann,
                                                    value=False)
        ann_modification_radio = ttk.Radiobutton(self.ann_frame, text="hyperspeed simulated annealing algorithm",
                                                 variable=mod_ann,
                                                 value=True)
        ann_no_modification_radio.pack(pady=5)
        ann_modification_radio.pack(pady=5)

        ann_confirm_button = ttk.Button(self.ann_frame, text="Confirm", command=self.ann_confirm_mod)
        ann_confirm_button.pack(pady=10)

        self.ann_hamilton_length_label = tk.Label(self.ann_frame, text="Cycle Length: ")
        self.ann_hamilton_length_label.pack()

        self.ann_hamilton_fig = Figure(figsize=(6, 6), dpi=100)
        self.ann_hamilton_ax = self.ann_hamilton_fig.add_subplot(111)
        self.ann_hamilton_canvas = FigureCanvasTkAgg(self.ann_hamilton_fig, master=self.ann_frame)
        self.ann_hamilton_canvas.draw()
        self.ann_hamilton_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ann_edge_weights = {}
        self.ann_mod_var = tk.BooleanVar()

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
    def clear_canvas(self, event=None):
        self.G.clear()
        self.pos.clear()
        self.history.clear()
        self.selected_node = None
        self.edge_weights.clear()
        self.update_plot()

    def nn_confirm_mod(self):
        global mod_nn
        mod_nn = self.nn_mod_var.get()

    def ann_confirm_mod(self):
        global mod_ann
        mod_ann = self.ann_mod_var.get()

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
    def nn_find_hamilton_cycle(self):
        if not self.G.nodes():  # just checkin' for empty cycle
            self.nn_hamilton_cycle_text.delete("1.0", tk.END)
            self.nn_hamilton_cycle_text.insert(tk.END, "Graph is empty!")
            self.nn_hamilton_length_label.config(text="Cycle Length: N/A")
            self.nn_update_hamilton_plot(None)
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
        hamilton_cycle, total_length = find_hamiltonian_cycle(self.G, mod=mod_nn)

        if hamilton_cycle:
            # display the result
            cycle_str = " -> ".join(hamilton_cycle)
            self.nn_hamilton_cycle_text.delete("1.0", tk.END)
            self.nn_hamilton_cycle_text.insert(tk.END, cycle_str)
            self.nn_hamilton_length_label.config(text=f"Cycle Length: {total_length:.2f}")  # update length label
            self.nn_update_hamilton_plot(hamilton_cycle)
        else:
            self.nn_hamilton_cycle_text.delete("1.0", tk.END)
            self.nn_hamilton_cycle_text.insert(tk.END, "No Hamilton cycle found.")
            self.nn_hamilton_length_label.config(text="Cycle Length: N/A")
            self.nn_update_hamilton_plot(None)

    def nn_update_hamilton_plot(self, path):
        self.nn_hamilton_ax.clear()
        self.nn_hamilton_ax.set_xlim(-1, 10)
        self.nn_hamilton_ax.set_ylim(-1, 10)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.nn_hamilton_ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.nn_hamilton_ax)

        if path:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=edges,
                ax=self.nn_hamilton_ax,
                edge_color='r',
                width=2,
                connectionstyle="arc3,rad=0.2",
                arrows=True,
                arrowstyle="-|>"
            )
        self.nn_hamilton_canvas.draw()

    # and also simulated annealing logic too
    def ann_find_hamilton_cycle(self, init_temp=1000, cooling_rate=0.999, iterations=5000, mod=True):
        if not self.G.nodes():
            self.ann_hamilton_cycle_text.delete("1.0", tk.END)
            self.ann_hamilton_cycle_text.insert(tk.END, "Graph is empty!")
            self.ann_hamilton_length_label.config(text="Cycle Length: N/A")
            self.ann_update_hamilton_plot(None)
            return

        # Start with a random cycle
        nodes = list(self.G.nodes())
        start_node = random.choice(nodes) if nodes else None
        initial_path, initial_cost = self.ann_find_init_path([start_node], 0, {})
        current_cycle = initial_path if initial_path else nodes + [nodes[0]]
        current_cost = initial_cost if initial_path else float('inf')

        # Initialize best_cycle and best_cost with the initial cycle
        best_cycle = current_cycle
        best_cost = current_cost
        if mod is False:
            for i in range(iterations):
                # Get a neighbor cycle by swapping two nodes
                neighbor_cycle = self.get_neighbor_cycle(current_cycle)
                neighbor_cost = self.calculate_cycle_cost(neighbor_cycle)

                # Calculate cost difference
                cost_diff = neighbor_cost - current_cost

                # Acceptance probability
                if cost_diff < 0 or random.random() < math.exp(-cost_diff / init_temp):
                    current_cycle = neighbor_cycle
                    current_cost = neighbor_cost

                # Update best cycle if current cycle is better
                if current_cost < best_cost:
                    best_cycle = current_cycle
                    best_cost = current_cost

                # Cool down
                init_temp *= cooling_rate
        else:
            beta = 0.1
            init_temp = init_temp
            for i in range(iterations):
                neighbor_cycle = self.get_neighbor_cycle(current_cycle)
                neighbor_cost = self.calculate_cycle_cost(neighbor_cycle)

                # Пропуск недопустимых циклов
                if neighbor_cost == float('inf'):
                    continue

                cost_diff = neighbor_cost - current_cost

                # Вероятность Лопатина
                if cost_diff < 0:
                    acceptance_prob = 1.0
                else:
                    acceptance_prob = 1 / (1 + math.exp(cost_diff / init_temp))

                if random.random() < acceptance_prob:
                    current_cycle = neighbor_cycle
                    current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_cycle = current_cycle
                    best_cost = current_cost

                init_temp = init_temp / (1 + beta * i)
                init_temp = max(init_temp, 1e-10)

        # Display the result after the loop finishes
        if best_cycle:
            cycle_str = " -> ".join(best_cycle)
            self.ann_hamilton_cycle_text.delete("1.0", tk.END)
            self.ann_hamilton_cycle_text.insert(tk.END, cycle_str)
            self.ann_hamilton_length_label.config(text=f"Cycle Length: {best_cost:.2f}")
            self.ann_update_hamilton_plot(best_cycle)
        else:
            self.ann_hamilton_cycle_text.delete("1.0", tk.END)
            self.ann_hamilton_cycle_text.insert(tk.END, "No Hamilton cycle found.")
            self.ann_hamilton_length_label.config(text="Cycle Length: N/A")
            self.ann_update_hamilton_plot(None)

    def ann_find_init_path(self, current_path, current_cost, cost_matrix):
        if len(current_path) == len(self.G.nodes()):
            first = current_path[0]
            last = current_path[-1]
            # Проверка направленности ребра
            if self.G.has_edge(last, first):  # Только прямое ребро
                return current_path + [first], current_cost + self.calculate_weight(last, first)
            return None, float('inf')

        last_node = current_path[-1]
        best_path, min_cost = None, float('inf')

        # Получение только прямых соседей
        neighbors = list(self.G.successors(last_node))  # Для направленных графов

        for neighbor in random.sample(neighbors, len(neighbors)):
            if neighbor not in current_path:
                new_cost = current_cost + self.calculate_weight(last_node, neighbor)
                path, total_cost = self.ann_find_init_path(current_path + [neighbor], new_cost, cost_matrix)
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = path
        return best_path, min_cost

    def get_neighbor_cycle(self, cycle):
        i, j = sorted(random.sample(range(1, len(cycle) - 1), 2))
        neighbor = cycle[:i] + cycle[i:j + 1][::-1] + cycle[j + 1:]

        for k in range(len(neighbor) - 1):
            u, v = neighbor[k], neighbor[k + 1]
            if not self.G.has_edge(u, v):  
                return cycle

        return neighbor

    def calculate_cycle_cost(self, cycle):
        total_cost = 0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            # Проверка только прямых ребер
            if (u, v) in self.edge_weights:
                total_cost += self.edge_weights[(u, v)]
            elif self.G.has_edge(u, v):  # Для графов без весов
                total_cost += self.calculate_weight(u, v)
            else:
                return float('inf')  # Направленное ребро отсутствует
        return total_cost

    def ann_update_hamilton_plot(self, path):
        self.ann_hamilton_ax.clear()
        self.ann_hamilton_ax.set_xlim(-1, 10)
        self.ann_hamilton_ax.set_ylim(-1, 10)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ann_hamilton_ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ann_hamilton_ax)

        if path:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=edges,
                ax=self.ann_hamilton_ax,
                edge_color='r',
                width=2,
                connectionstyle="arc3,rad=0.2",
                arrows=True,
                arrowstyle="-|>"
            )
        self.ann_hamilton_canvas.draw()

    def run(self):
        self.root.mainloop()


mod_ann = False
mod_nn = False
plotter = DirectedGraphPlotter()
plotter.run()
