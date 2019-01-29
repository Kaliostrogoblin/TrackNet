import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm

hitspath = "/Users/egor/prog/dubna/TrackNet/bmn_tracking/bmn_box_default_10K_events_hits.csv"
mcpath = "/Users/egor/prog/dubna/TrackNet/bmn_tracking/bmn_box_default_10K_events_mc_points.csv"


def addStationIndex(df, column_to_index, stationCount = 6):
    # rounding to the nearest 10 by z coordinate, and assuring that we will have 6 groups
    grouping = df.groupby(df[column_to_index].apply(lambda x: round(round(x) / 10)))
    assert len(grouping.groups) == 6, "STATIONS COUNT SHOULD BE 6!!"
    df['station_id'] = grouping.ngroup()
    return df

def main(validate = False):

    mc_df = pd.read_csv(mcpath, encoding="utf-8")
    mc_stations = addStationIndex(mc_df, 'z_in')
    mc_grouped_event_station = mc_stations.groupby(['station_id', 'event_id'])

    hits_df = pd.read_csv(hitspath, encoding="utf-8")
    hits_grouped_event_station = hits_df.groupby(['station_id', 'event_id'])
    result = []
    for (name, group) in hits_grouped_event_station:
        print(name)
        x, y = group["x"].to_numpy(), group["y"].to_numpy()
        if validate:
            plt.plot(x,y)
            plt.show()
            return
        hits_points = np.column_stack((x, y))
        tree = cKDTree(hits_points)

        mc_points_cur_event_cur_station_group = mc_grouped_event_station.get_group(name)
        mc_x, mc_y = (mc_points_cur_event_cur_station_group["x_in"].to_numpy(),
                                 mc_points_cur_event_cur_station_group["y_in"].to_numpy())
        mc_xy = np.column_stack((mc_x, mc_y))
        print(mc_xy)
        found_dist, found_idx = tree.query(mc_xy)
        print(found_idx)
        ax1 = plt.subplot(311)
        ax1.set_xlim(9.0, 10)
        plt.plot(x, y, 'ro', label='hits')
        ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
        colors = cm.rainbow(np.linspace(0, 1, len(mc_x)))
        for num, idx in enumerate(mc_x):
            plt.scatter(mc_x[num], mc_y[num], s=35 , c = colors[num])
        for i, txt in enumerate(mc_x):
            ax2.annotate(i, (mc_x[i], mc_y[i]))
        plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.plot(x, y,'ro', mc_x, mc_y, 'bo')
        plt.show()
        print("HITS                            MC POINTS")
        for num, idx in enumerate(found_idx):
            print(hits_points[idx], mc_xy[num])
        return
        tree.query()


if __name__ == '__main__':
    main()
