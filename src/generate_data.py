import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import pickle

def save_data(data):
    with open('data/successful_nets.pkl', 'wb') as f:
            pickle.dump(data, f)
    print(f"Saved {len(successful_nets)} successful configurations.")

def matrix_rank(switch_matrix, incidence_matrix):
    total_nodes = len(net.bus.index)
    # Calculate incidence matrix - switch Matrix dot product
    AS_matrix = np.dot(incidence_matrix, switch_matrix)
    # Calculate rank of the matrix
    rank = np.linalg.matrix_rank(AS_matrix)
    # print(rank)
    if rank == total_nodes - 1:
        print('All nodes are connected')
        return True

    else:
        print('Not all nodes are connected')
        return False

def matrix_trace(switch_matrix):
    total_nodes = len(net.bus.index)
    trace = np.trace(switch_matrix)
    if trace == total_nodes - 1:
        print('Network is radial')       
        return True
    else:
        print('Network is NOT radial')       
        return False


def plot_graph_network(net):
    graph = pp.topology.create_nxgraph(net)
    pos = nx.spring_layout(graph, k=1, iterations=1000)  # Increase k for more spread
    plt.figure(figsize=(10, 6))  # Width, height in inches

    # Check if the geodata DataFrame exists and contains x, y coordinates
    # if 'bus_geodata' in net and 'x' in net.bus_geodata and 'y' in net.bus_geodata:
    #     # Scale and adjust positions to make them more spread out and structured
    #     pos = {bus: (x * 3, y * 3) for bus, x, y in zip(net.bus_geodata.index, net.bus_geodata.x, net.bus_geodata.y)}
    # else:
    #     pos = nx.spring_layout(graph, scale=2.0, center=(0.5, 0.5))  # Adjust scale and center as needed
 
    nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_size=8, font_color='white')
    plt.title(f'33 bus system (Network {len(successful_nets)})')
    plt.savefig(f'plots/Network_{len(successful_nets)}', dpi=300)
    plt.show()
    return

def incidence_matrix(net):
    new_net = copy.deepcopy(net)
    mask = ~new_net.line['in_service']
    new_net.line.loc[mask, 'in_service'] = True
    lines = new_net.line
    nodes = new_net.bus.index
    # Initialize adjacency matrix with zeros
    incidence_matrix = np.zeros((len(nodes), len(lines)))
    # Add edges from the lines in the network
    for idx, line in net.line.iterrows():
        incidence_matrix[line.from_bus, idx] = -1
        incidence_matrix[line.to_bus, idx] = 1
    return incidence_matrix

def service_matrix(net):
    switch_matrix = np.zeros((len(net.line), len(net.line)))
    in_service = net.line['in_service']
    np.fill_diagonal(switch_matrix, in_service)
    return switch_matrix

def new_network_configuration(net):
    selected_nodes = np.random.choice(net.line.index[1:], size=5)
    mask = ~net.line['in_service']
    net.line.loc[mask, 'in_service'] = True
    net.line.loc[selected_nodes, 'in_service'] = False
    return net

if __name__ == '__main__':
    # Initialize script
    net = nw.case33bw()
    switch_matrix = service_matrix(net)
    incidence_matrix = incidence_matrix(net)
    successful_nets = []

    # Load file with load factors per season
    load_factors = pd.read_csv('data/load_seasons.csv')

    while len(successful_nets) < 10:
        try: 
            net = new_network_configuration(net)
            switch_matrix = service_matrix(net)
            if matrix_rank(switch_matrix, incidence_matrix) and matrix_trace(switch_matrix):
                pp.runpp(net, verbose=False, numba=False)
            else:
                print("Criteria for radial network and convergence not met.")
                continue  # Skip the rest of the loop and proceed with the next iteration

        except pp.LoadflowNotConverged:
            print('Load flow did NOT converge.')

        else:
            successful_nets.append(deepcopy(net))
            plot_graph_network(net)
            print('Load flow converged')

    save_data(successful_nets)

