import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt

net = nw.case33bw()


def run_opf(network):
    pp.runopp(network, verbose=False, numba=False)  # Run the optimal power flow
    return network.res_bus, network.res_line

def save_data(bus_data, line_data):
    data = np.hstack((bus_data.to_numpy(), line_data.to_numpy()))
    # data.tonumpy()
    data.save('dataset_33bus')
    return

def is_radial(incidence_matrix):
    n = incidence_matrix.shape[0]
    # Calculate rank of the matrix
    rank = np.linalg.matrix_rank(incidence_matrix)
    print(rank)
    if rank == n - 1:
        return True

    return False

def self_loop(net):
    adjacency_matrix = np.zeros((len(net.bus.index), len(net.bus.index)))
    for _, lines in net.line.iterrows():
        f, t = lines['from_bus'], lines['to_bus']
        adjacency_matrix[f-1][t-1] = 1
        adjacency_matrix[t-1][f-1] = 1
    # determine if the matrix has loops
    trace = np.trace(adjacency_matrix)
    if trace > 0:
        return True

    return False


def plot_graph_network(net):
    # pos = nx.spring_layout(graph)  # positions for all nodes
    graph = pp.topology.create_nxgraph(net)
    pos = nx.spring_layout(graph, k=1, iterations=1000)  # Increase k for more spread
    nx.draw_networkx(graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=8, font_color='darkred')
    plt.title('33 bus network')
    plt.show()
    return



def get_incidence_matrix(net):
    nodes = net.bus.index
    lines = net.line
    
    # Initialize adjacency matrix with zeros
    incidence_matrix = np.zeros((len(nodes), len(lines)))

    # Add edges from the lines in the network
    for idx, line in net.line.iterrows():
        incidence_matrix[line.from_bus, idx] = 1
        incidence_matrix[line.to_bus, idx] = -1

    return incidence_matrix

def new_network_connection(net):
    for i in range(10):
        original_state = net.line.copy()
        mask = np.where(net.line['in_service'] == False)
        net.line['in_service'].iloc[mask] = True
        selected_nodes = np.random.choice(net.line.index[1:], size=5)
        net.line['in_service'].iloc[selected_nodes] = False
        plot_graph_network(net)
        # if not self_loop(net) and is_radial(incidence_matrix):
        #     x = np.append(np.array(net.line['in_service']))

    return net

if __name__ == '__main__':
    incidence_matrix = get_incidence_matrix(net)
    # switch_matrix = switch_matrix(net)

    new_network_connection(net)
    
    if not self_loop(net) and is_radial(incidence_matrix):
        bus_data, line_data = run_opf(net)
        plot_graph_network(net)
