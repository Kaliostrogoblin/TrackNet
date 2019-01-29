import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm

hitspath = "/Users/egor/prog/dubna/TrackNet/bmn_tracking/bmn_box_default_10K_events_hits.csv"
mcpath = "/Users/egor/prog/dubna/TrackNet/bmn_tracking/bmn_box_default_10K_events_mc_points.csv"


def addStationIndex(df, column_to_index, stationCount=6):
    # rounding to the nearest 10 by z coordinate, and assuring that we will have 6 groups
    grouping = df.groupby(df[column_to_index].apply(lambda x: round(round(x) / 10)))
    assert len(grouping.groups) == 6, "STATIONS COUNT SHOULD BE 6!!"
    df['station_id'] = grouping.ngroup()
    return df


def main(validate=False):
    mc_df = pd.read_csv(mcpath, encoding="utf-8")
    mc_stations = addStationIndex(mc_df, 'z_in')

    mc_filtered_grouped = mc_stations.groupby(['event_id', 'track_id'], as_index=False) \
        .filter(lambda x: len(x['track_id']) > 1) \
        .groupby(['station_id', 'event_id'])
    # mc_grouped = mc_stations.groupby(['event_id']).filter(lambda df: df.track_id.value_counts() > 1)

    hits_df = pd.read_csv(hitspath, encoding="utf-8")
    hits_grouped_event_station = hits_df.groupby(['station_id', 'event_id'], as_index=False)
    hits_df['track_id'] = -1
    result = []
    count = 0
    for (name, group) in hits_grouped_event_station:
        print(name, end='\r', flush=True)
        # print(group.index[0])
        x, y = group["x"].to_numpy(), group["y"].to_numpy()
        if validate:
            plt.plot(x, y)
            plt.show()
            return
        hits_points = group[['x', 'y']].values
        tree = cKDTree(hits_points)

        mc_group = mc_filtered_grouped.get_group(name).reset_index()
        # print(mc_group)

        mc_xy = mc_group[['x_in', 'y_in']].values
        mc_x, mc_y = (mc_group["x_in"].to_numpy(),
                      mc_group["y_in"].to_numpy())
        found_dist, found_idx = tree.query(mc_xy)
        found_idx_set = set(found_idx)
        # print(found_idx)
        if len(found_idx_set) != len(mc_xy):
            validate_draw(found_idx,hits_points, mc_x, mc_xy, mc_y,x,y)
            raise Exception("Error. More than one mc point belong to hit. event_id", name[1])
        # or mc_group.track_id.iloc[found_idx[x[0]]] if x[0] in found_idx else -1
        for i, idx in enumerate(found_idx):
            # print("[", )
            # print(hits_df.at[group.index[idx], 'x'], hits_df.at[group.index[idx], 'y'], "] :(", )
            # print(mc_group.x_in.iloc[i], mc_group.y_in.iloc[i], ")")
            if hits_df.at[group.index[idx], 'track_id'] != -1:
                raise Exception("Error. trying to set existing track id")
            hits_df.at[group.index[idx], 'track_id'] = mc_group.track_id.iloc[i]
        # for val,dat in group.group():
        #     dat['track_id'] = 10
        # print("\n\n==========================\n\n", group)
        # validate_draw(found_idx, hits_points, mc_x, mc_xy, mc_y, x, y)
        # tree.query()
    print(hits_df)


def validate_draw(found_idx, hits_points, mc_x, mc_xy, mc_y, x, y):
    ax1 = plt.subplot(311)
    plt.plot(x, y, 'ro', label='hits')
    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    colors = cm.rainbow(np.linspace(0, 1, len(mc_x)))
    for num, idx in enumerate(mc_x):
        plt.scatter(mc_x[num], mc_y[num], s=35, c=colors[num])
    for i, txt in enumerate(mc_x):
        ax2.annotate(i, (mc_x[i], mc_y[i]))
    plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(x, y, 'ro', mc_x, mc_y, 'bo')
    plt.show()
    print("HITS                            MC POINTS")
    for num, idx in enumerate(found_idx):
        print(hits_points[idx], mc_xy[num])



if __name__ == '__main__':
    main()
