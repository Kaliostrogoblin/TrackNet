import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import sys

debug = False
show_progress = True


def progress(count, total, status=''):
    if not show_progress:
        return
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


hitspath = "bmn_tracking/bmn_box_default_10K_events_hits.csv"
mcpath = "bmn_tracking/bmn_box_default_10K_events_mc_points.csv"
matchedpath = "bmn_tracking/kek.csv"


def addStationIndex(df, column_to_index, stationCount=6):
    # rounding to the nearest 10 by z coordinate, and assuring that we will have 6 groups
    grouping = df.groupby(df[column_to_index].apply(lambda x: round(round(x) / 10)))
    assert len(grouping.groups) == 6, "STATIONS COUNT SHOULD BE 6!!"
    df['station_id'] = grouping.ngroup()
    return df


def main(validate=False):
    mc_df = pd.read_csv(mcpath, encoding="utf-8")

    mc_stations = addStationIndex(mc_df, 'z_in')
    total_events = len(mc_stations.event_id.value_counts())

    mc_filtered_grouped = mc_stations.groupby(['event_id', 'track_id'], as_index=False) \
        .filter(lambda x: len(x['track_id']) > 1 and progress(x.name[0], total_events, " filtering events.") is None) \
        .groupby(['station_id', 'event_id'])
    print("\nStarting event matching\n")
    # mc_grouped = mc_stations.groupby(['event_id']).filter(lambda df: df.track_id.value_counts() > 1)

    hits_df = pd.read_csv(hitspath, encoding="utf-8")
    hits_grouped_event_station = hits_df.groupby(['station_id', 'event_id'], as_index=False)
    hits_df['track_id'] = -1

    for (name, group) in hits_grouped_event_station:
        progress(name[1], total_events, " matching events for station #%4d." % name[0])
        hits_points = group[['x', 'y']].values
        tree = cKDTree(hits_points)
        try:
            mc_group = mc_filtered_grouped.get_group(name).reset_index()
        except KeyError:
            if debug:
                print("Warning, event_id #", name[1], "was not found in mc_points.")
            continue

        mc_xy = mc_group[['x_in', 'y_in']].values
        found_dist, found_idx = tree.query(mc_xy)
        found_idx_set = set(found_idx)
        group_index = group.index.values
        if len(found_idx_set) != len(mc_xy):
            unique_set = set()
            for i in range(len(found_idx)):
                if found_idx[i] in unique_set:
                    if debug:
                        print("Warning. Doubling row #", group.index[found_idx[i]],
                              " because more than 1 mc point was found for this hit.")
                    hits_df.loc[len(hits_df)] = hits_df.loc[group.index[found_idx[i]]]
                    group_index = np.append(group_index, (len(hits_df) - 1))
                    found_idx[i] = len(group_index) - 1
                else:
                    unique_set.add(found_idx[i])
            if validate:
                raise Exception("Error. More than one mc point belong to hit. event_id", name[1])
        for i, idx in enumerate(found_idx):
            if hits_df.at[group_index[idx], 'track_id'] != -1:
                raise Exception("Error. trying to set existing track id")
            hits_df.at[group_index[idx], 'track_id'] = mc_group.track_id.iloc[i]

    hits_df = hits_df.astype({'event_id': int, 'station_id': int, 'track_id': int})
    hits_df.to_csv("kek.csv", index=None)


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

def dist(df_row):
    a = df_row['x':'z'].values
    b = df_row['x_in':'z_in'].values
    progress(df_row.event_id, dist.total_events, " validating events.")
    return np.linalg.norm(a - b)

dist.total_events = 0


def validate_all():
    print('VALIDATE')
    mc_df = pd.read_csv(mcpath, encoding="utf-8")
    mc_stations = addStationIndex(mc_df, 'z_in')
    matched_df = pd.read_csv(matchedpath, encoding="utf-8")
    print('MERGING....')
    res = pd.merge(matched_df, mc_stations, on=['event_id', 'track_id', 'station_id'])
    print('COUNTING....\n')
    total_events = len(mc_stations.event_id.value_counts())
    dist.total_events = total_events
    t = res.apply(dist, axis=1).describe()
    print('\nRESULT:')
    print(t)
    t.to_csv('RESULT.csv', header=True)
    return

if __name__ == '__main__':
    validate_all()
