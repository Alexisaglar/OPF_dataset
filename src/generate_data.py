import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
import copy

def run_opf(network):
    pp.runopp(network, verbose=False, numba=False)  # Run the optimal power flow
    return network.res_bus, network.res_line

def save_data(bus_data, line_data):
    # data = np.hstack((bus_data.to_numpy(), line_data.to_numpy()))
    # data.tonumpy()
    # data.save('dataset_33bus')
    return


def matrix_rank(switch_matrix, incidence_matrix):
    total_nodes = len(net.bus.index)

    # Calculate incidence matrix - switch Matrix dot product
    AS_matrix = np.dot(incidence_matrix, switch_matrix)
    # Calculate rank of the matrix
    rank = np.linalg.matrix_rank(AS_matrix)
    print(rank)
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
    # pos = nx.spring_layout(graph)  # positions for all nodes
    graph = pp.topology.create_nxgraph(net)
    pos = nx.spring_layout(graph, k=1, iterations=1000)  # Increase k for more spread
    nx.draw_networkx(graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=8, font_color='darkred')
    plt.title('33 bus network')
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
    net = nw.case33bw()
    switch_matrix = service_matrix(net)
    incidence_matrix = incidence_matrix(net)
    for i in range(1000):
        net = new_network_configuration(net)
        print(net.line['in_service'])
        switch_matrix = service_matrix(net)
        if matrix_rank(switch_matrix, incidence_matrix):
            if matrix_trace(switch_matrix):
                plot_graph_network(net)
        
