import random
import time
import networkx as nx
import numpy as np


def generate_graph_for_test(n=100, ):
    G = nx.erdos_renyi_graph(n, 0.5, directed=True)

    normal_weights = np.abs(np.random.normal(0, 1, len(G.edges())))
    nx.set_edge_attributes(G, dict(zip(G.edges(), normal_weights)), "weight")
    return G


def find_hamiltonian_cycle(graph, mod=True):
    def find_path(current_path, current_cost):
        if len(current_path) == len(graph.nodes()):  # this is how we end the path finding
            first_node = current_path[0]
            last_node = current_path[-1]
            if graph.has_edge(last_node, first_node):
                cost = graph[last_node][first_node]["weight"]
                return current_path + [first_node], current_cost + cost
            else:
                return None, float('inf')

        last_node = current_path[-1]
        best_path = None
        min_cost = float('inf')
        for neighbor in graph.neighbors(last_node):
            if neighbor not in current_path:
                cost = graph[last_node][neighbor]["weight"]
                path, total_cost = find_path(current_path + [neighbor], current_cost + cost)
                if path and total_cost < min_cost:
                    min_cost = total_cost
                    best_path = path

        return best_path, min_cost

    best_cycle = None
    min_cycle_cost = float('inf')
    if mod is True:
        start_time = time.time()
        for start_node in graph.nodes():
            cycle, cycle_cost = find_path([start_node], 0)
            if cycle and cycle_cost < min_cycle_cost:
                min_cycle_cost = cycle_cost
                best_cycle = cycle
        end_time = time.time()
        time_diff = end_time - start_time
    else:
        start_time = time.time()
        start_node = random.choice(list(graph.nodes()))
        best_cycle, min_cycle_cost = find_path([start_node], 0)
        end_time = time.time()
        time_diff = end_time - start_time
    return best_cycle, min_cycle_cost, time_diff


def compare_win_results(arr1, arr2):
    results = [0] * 100
    for i in range(len(arr1)):
        if arr1[i][1] > arr2[i][1]:
            results[i] = 1
        elif arr1[i][1] < arr2[i][1]:
            results[i] = -1
        else:
            results[i] = 0
    return results


def compare_time_results(time_results):
    time_diff = []
    time_diff_times = []
    for time_comp in time_results:
        time_diff.append(time_comp[1] - time_comp[0])
        if time_comp[0] != 0:
            time_diff_times.append(time_comp[1] * 1000 / (time_comp[0] * 1000))
        else:
            print(
                "zero division! just skippin' it")  # it happens because sometimes when we choose random node, there is a small chance that this node has no edges at all
    return round(np.mean(time_diff), 7), round(np.mean(time_diff_times), 2)


def print_results(win_results: list[int], no_ham: int, time_results, n_nodes):
    win_diff = sum(win_results)
    mod_wins = win_results.count(1)
    no_mod_wins = win_results.count(-1)
    draws = win_results.count(0)
    time_diff, time_diff_times = compare_time_results(time_results)
    if win_diff > 0:
        print(
            f"modded variant({n_nodes} nodes) wins more often({mod_wins} vs {no_mod_wins}) "
            f"and there is also {draws} draws "
            f"(and this algorithm couldnt find hamilton cycle in {no_ham} cases)"
            f"but modded variant was {time_diff_times} times slower (mean) then the classic heuristic"
            f"({time_diff} seconds slower (mean)")
    else:
        print(
            f"both variants win with the same frequency({no_mod_wins}) "
            f"and there is also {draws} draws "
            f"(and this algorithm couldnt find hamilton cycle in {no_ham} cases)")


def main(n_iter=100, n_nodes=10):
    results_no_mod = []
    results_mod = []
    time_results = []
    no_ham = 0
    for graph_num in range(n_iter):
        G = generate_graph_for_test(n_nodes)
        print(f"graph {graph_num + 1} was generated")
        hamilton_cycle_no_mod, total_length_no_mod, time_no_mod = find_hamiltonian_cycle(G, mod=False)
        print(f"no_mod time spent: {time_no_mod}")
        results_no_mod.append((hamilton_cycle_no_mod, total_length_no_mod))
        hamilton_cycle_mod, total_length_mod, time_mod = find_hamiltonian_cycle(G, mod=True)
        print(f"mod time spent: {time_mod}")
        results_mod.append((hamilton_cycle_mod, total_length_mod))

        time_results.append((time_no_mod, time_mod))
        if total_length_mod == float("inf"):
            no_ham += 1

    win_results = compare_win_results(results_no_mod,
                                      results_mod)  # array that shows the amount of better solutions by modded version
    print_results(win_results, no_ham, time_results, n_nodes)


num_of_nodes = [9, 10, 11, 12]
for i in num_of_nodes:
    main(100, i)
