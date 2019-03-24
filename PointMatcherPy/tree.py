import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

all_path = 'bmn_tracking/CC4GeVmb_100_n50k.tsv'

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
    all_df = pd.read_csv(all_path, encoding="utf-8", delimiter='\t', nrows=2000)
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

        return res_adj_list
    else:
        by_stations = [df.rename(columns={'station': 'station_' + str(ind)}) for (ind, df) in
                       event_df.groupby('station')]
        for i in range(1, len(by_stations)):
            cartesian_product = pd.merge(by_stations[i - 1], by_stations[i], on='event')
            elems = [(row.index_old_x, row.index_old_y, dist_tuple(row)) for row in cartesian_product.itertuples()]
            res_adj_list.extend(elems)

        return res_adj_list


def visualize_graph(G):
    plt.figure()
    pos = nx.spring_layout(G)  # positions for all nodes
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=1)
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=2, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

    plt.axis('off')

def tree_try():
    G = nx.Graph()
    event_id = 1

    # get graph for all hits
    all = get_adj_nodes_dist_all_for_event_id(event_id, True)
    G.add_weighted_edges_from(all)
    G = nx.minimum_spanning_tree(G)
    visualize_graph(G)

    # get graph for only true hits
    true_only = get_adj_nodes_dist_all_for_event_id(event_id, False)
    G1 = nx.Graph()
    G1.add_weighted_edges_from(true_only)
    visualize_graph(G1)

    plt.show()
    pass


if __name__ == '__main__':
    tree_try()
