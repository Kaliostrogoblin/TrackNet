import matplotlib

matplotlib.use('Qt5Agg')

from argparse import ArgumentParser
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.widgets import TextBox, Button
import sys
from collections import namedtuple
import functools
import networkx as nx

all_path = 'bmn_tracking/CC4GeVmb_100_n50k.tsv'


def namedtuple_me(s, name='PNT'):
    return namedtuple(name, s.index)(*s)


def addStationIndex(df, column_to_index, stationCount=6):
    # rounding to the nearest 10 by z coordinate, and assuring that we will have 6 groups
    grouping = df.groupby(df[column_to_index].apply(lambda x: round(round(x) / 10)))
    assert len(grouping.groups) == 6, "STATIONS COUNT SHOULD BE 6!!"
    df['station'] = grouping.ngroup()
    return df


def dist(df_row1, df_row2):
    a = df_row1['x':'y'].values
    b = df_row2['x':'y'].values
    return np.linalg.norm(a - b)


def dist_tuple(row):
    a = np.array([row.x_x, row.y_x])
    b = np.array([row.x_y, row.y_y])
    return np.linalg.norm(a - b)


def get_adj_nodes_dist_all_for_event_id(event_id, preserve_fakes):
    all_df = pd.read_csv(all_path, encoding="utf-8", delimiter='\t', nrows=10000)
    all_df = addStationIndex(all_df, 'z')
    all_df['index_old'] = all_df.index
    event_df = all_df[all_df['event'] == event_id]

    # prefiltering events
    event_df = event_df.groupby('track', as_index=False).filter(
        lambda x: x.station.nunique() == 6 and x.station.value_counts().max() == 1 or
                  preserve_fakes and x.track.values[0] == -1
        # if preserve_fakes == False, we are leaving only matched events, no fakes
    )
    res_adj_list = []
    if not preserve_fakes:
        grouped = event_df.groupby(['track'])
        for (ind, group) in grouped:
            for row in range(1, len(group.index)):
                elem = (group.index[row - 1], group.index[row], dist(group.iloc[row - 1], group.iloc[row]))

                res_adj_list.append(elem)

        return res_adj_list, all_df
    else:
        by_stations = [df.rename(columns={'station': 'station_' + str(ind)}) for (ind, df) in
                       event_df.groupby('station')]
        for i in range(1, len(by_stations)):
            cartesian_product = pd.merge(by_stations[i - 1], by_stations[i], on='event')
            elems = [(row.index_old_x, row.index_old_y, dist_tuple(row)) for row in cartesian_product.itertuples()]
            res_adj_list.extend(elems)

        return res_adj_list, all_df


# accepts array of tuples -- (graph, pandas_data)
def prepare_graph_data(graph_with_pd):
    G, pd_data = graph_with_pd
    edges_data = G.edges(data=True)
    line_array_to_draw = []
    for (u, v, d) in edges_data:
        a = namedtuple_me(pd_data.iloc[u])
        b = namedtuple_me(pd_data.iloc[v])
        elem = (a, b, d)
        line_array_to_draw.append(elem)
    return line_array_to_draw
    #
    # pos = nx.spring_layout(G)  # positions for all nodes
    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    # # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=50)
    #
    # # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge,
    #                        width=1)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                        width=2, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')
    #
    # plt.axis('off')


class Visualizer:

    def __init__(self, all_data):
        self.__all_data = all_data
        self.__axs = []
        self.__color_map = {-1: (0.1, 0.1, 0.1)}

    def init_draw(self):
        matplotlib.rcParams['legend.fontsize'] = 10
        for single_data, with_edges, with_pnts, title in self.__all_data:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title)
            ax.set_xlabel('Station')
            ax.set_ylabel('X')
            ax.set_zlabel('Y')
            legends = {}
            for (fr_p, to_p, dist) in single_data:
                color, label, tr_id = self.generate_color_label(fr_p.track, to_p.track)
                if with_edges:
                    ax.plot((fr_p.station, to_p.station), (fr_p.x, to_p.x), zs=(fr_p.y, to_p.y),
                            c=color)
                marker_1 = 'h' if fr_p.track == -1 else 'o'
                marker_2 = 'h' if to_p.track == -1 else 'o'
                if with_pnts:
                    ax.scatter(fr_p.station, fr_p.x, fr_p.y, c =self.__color_map[int(fr_p.track)], marker=marker_1)
                    ax.scatter(to_p.station, to_p.x, to_p.y, c =self.__color_map[int(to_p.track)], marker=marker_2)
                if int(tr_id) not in legends:
                    legends[int(tr_id)] = mpatches.Patch(color=color, label=label)
            fig.legend(handles=list(legends.values()))

        plt.draw_all()
        pass

    def redraw_all(self):
        pass

    def generate_color_label(self, tr_id_from, tr_id_to):
        if tr_id_from not in self.__color_map:
            self.__color_map[tr_id_from] = np.random.rand(3, )
        if tr_id_to not in self.__color_map:
            self.__color_map[tr_id_to] = np.random.rand(3, )
        if tr_id_from != tr_id_to or tr_id_from == -1:
            return (0.1, 0.1, 0.1), 'fake connection', -1
        return self.__color_map[tr_id_from], 'tr_id: ' + str(int(tr_id_from)), tr_id_from


def tree_try():
    event_id = 27

    G = nx.Graph()
    # get graph for all hits
    all, df = get_adj_nodes_dist_all_for_event_id(event_id, True)
    G.add_weighted_edges_from(all)
    G = nx.minimum_spanning_tree(G)
    graph_data = prepare_graph_data((G, df))
    #
    #
    # get graph for only true hits
    true_only, df1 = get_adj_nodes_dist_all_for_event_id(event_id, False)
    G1 = nx.Graph()
    G1.add_weighted_edges_from(true_only)
    graph_data1 = prepare_graph_data((G1, df1))

    G2 = nx.Graph()
    # get graph for all hits
    all1, df2 = get_adj_nodes_dist_all_for_event_id(event_id, True)
    G2.add_weighted_edges_from(all)
    graph_data2 = prepare_graph_data((G2, df2))


    v = Visualizer([(graph_data, True, True, 'MINIMIZED GRAPH'),
                    (graph_data1, True, True, 'FULL TRACKS'),
                    (graph_data2, False, True, 'ALL EVENT HITS')])
    v.init_draw()
    # # G = nx.minimum_spanning_tree(G)

    # visualize_graph([(G, df), (G1, df1), (G2, df2)])
    plt.show()
    pass


if __name__ == '__main__':
    tree_try()
