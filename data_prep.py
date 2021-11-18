import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import matplotlib.pyplot as plt
import copy

import matplotlib.animation as animation
#%%
path = r"C:\Users\gupye\OneDrive\바탕 화면\뉴스레터\202103월호"

#%%
a = pd.read_csv(path + '\enr_cour_reg_202102.txt', sep = "|", encoding = 'utf8')
#%%
a
#%%


def nx_generate(a, schl_year, cut, filename):
    """
    calculate cosine similarity --> generate network
    node : students
    edge : similarity > cut

    :param a: dataframe
    :param schl_year: school year
    :param cut: smilarity cut
    :param filename: for saving
    :return: network
    """
    a = a[a['SCHL_YR'].isin(schl_year)]
    a['values'] = 1
    a_piv = pd.pivot_table(a, index = 'STD_ID', columns = 'COUR_CD', values = 'values', aggfunc=np.sum).fillna(0)
    a_sparse = sparse.csr_matrix(a_piv)
    similarities = cosine_similarity(a_sparse, dense_output=False)
    print(similarities)
    matrix = pd.DataFrame.sparse.from_spmatrix(similarities)

    mat = np.where(matrix.values<cut, 0, matrix.values) # 0.5이상 유사도만 엣지 살려둠

    #nodelist
    nodelist = a[['STD_ID', 'SCHL_YR', 'COL_NM', 'DEPT_NM','SEC_MAJOR_TP', 'SEC_MAJOR']].drop_duplicates().reset_index(drop = True).T.to_dict()

    #create a network
    B = nx.from_numpy_array(mat, parallel_edges = True, create_using=nx.MultiGraph())
    B.remove_edges_from(list(nx.selfloop_edges(B))) #self-loop 삭제
    nx.set_node_attributes(B, nodelist)

    print(B.nodes(data=True))
    print(B.edges(data=True))
    nx.write_gexf(B, path+filename)
    print('####Gephi saved####')
    return B
#%%
B_34_07 = nx_generate(a, ['3','4'], 0.5, '/newletter_mar_34_0.5.gexf')
#%%
from community import community_louvain as lvcm

""" Louvain method """
partition = lvcm.best_partition(graph=B_34_07, partition=None, weight='weight', resolution=1., randomize=True)
max_k_w = []
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    max_k_w = max_k_w + [list_nodes]
#%%
""" Make Community Color list """
community_num_group = len(max_k_w)
color_list_community = [[] for i in range(len(B_34_07.nodes()))]
for i in range(len(B_34_07.nodes())):
    for j in range(community_num_group):
        if i in max_k_w[j]:
            color_list_community[i] = j

#%%
community_num_group
#%%
""" Plot Community """
fig = plt.figure()
edges = B_34_07.edges(data=True)
pos = nx.spring_layout(B_34_07)
#weights = [B_34_07[u][v]['weight'] for u, v in edges]
Feature_color_sub = color_list_community
node_size = 50
im = nx.draw_networkx_nodes(B_34_07, pos, node_size=node_size, node_color=Feature_color_sub, cmap='jet', vmin=0, vmax=community_num_group)
nx.draw_networkx_edges(B_34_07, pos)
nx.draw_networkx_labels(B_34_07, pos, font_size=5, font_color="black")
plt.xticks([])
plt.yticks([])
plt.colorbar(im)
plt.show(block=False)
