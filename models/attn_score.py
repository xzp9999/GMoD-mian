import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的示例图
G = nx.DiGraph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# 假设这是你计算的注意力分数
attention_scores = [0.2, 0.4, 0.6, 0.8]

# 将注意力分数映射到边的颜色
edge_colors = attention_scores

# 可视化图
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues)

# 创建一个颜色条轴
cax = plt.axes([0.85, 0.1, 0.03, 0.65])  # 调整位置和大小
colorbar = plt.colorbar(edges, cax=cax, label="Attention Score")

plt.show()
