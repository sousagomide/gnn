import networkx as nx
import matplotlib.pyplot as plt

# g = nx.Graph()
# g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G')])
# nx.draw(g, with_labels=True, node_color='lightblue', edge_color='black', node_size=2000, font_size=16)
# plt.show()

# dg = nx.DiGraph()
# dg.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G')])
# nx.draw(dg, with_labels=True, node_color='lightblue', edge_color='black', node_size=2000, font_size=16)
# plt.show()

wg = nx.Graph()
wg.add_edges_from([
    ('A', 'B', {'weight': 10}), 
    ('A', 'C', {'weight': 20}), 
    ('B', 'D', {'weight': 30}), 
    ('B', 'E', {'weight': 40}), 
    ('C', 'F', {'weight': 50}), 
    ('C', 'G', {'weight': 60})])
pos = nx.spring_layout(wg)
nx.draw(wg, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=1000, font_size=16)
labels = nx.get_edge_attributes(wg, 'weight')
nx.draw_networkx_edge_labels(wg, pos, edge_labels=labels)
plt.show()