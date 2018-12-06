import numpy as np
import shutil
import os


def read_data(dirpath):
    x_train = np.load(os.path.join(dirpath, "X_train_all.npy"))
    x_test = np.load(os.path.join(dirpath, "X_test_all.npy"))
    y_train = np.load(os.path.join(dirpath, "y_train_all.npy"))
    y_test = np.load(os.path.join(dirpath, "y_test_all.npy"))
    return (x_train, y_train), (x_test, y_test)


def shuffle_arrays(*args):
    '''args - arrays of the same size'''
    assert len(set(len(x) for x in args)) == 1, 'Arrays must be the same size'
    idx = np.random.permutation(len(args[0]))
    args = [x[idx] for x in args]
    return args


def get_part(x, y, n_points):
    """Takes parts of x and y and create part with
        n_points - number of points track consists of
    """
    batch_xs = np.zeros_like(x)
    batch_ys = np.zeros((len(y), 3)) 
    # :2 - we take only (x, y) coords
    next_points = x[:, n_points, :2] 
    # [probability, x_coord, y_coord]
    batch_ys = np.concatenate([y, next_points], axis=1)
    # cut to n_points
    batch_xs[:, :n_points] = x[:, :n_points]
    return batch_xs, batch_ys
    

def get_dataset(x, y, min_points, shuffle=True):
    # from vector to matrix
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    # parts with different number of points
    n_parts = x.shape[1] - min_points
    # size of each part of dataset 
    part_size = len(x) // n_parts 
    # create dataset variables
    x_t = np.zeros_like(x[:n_parts*part_size])
    y_t = np.zeros((n_parts*part_size, 3))
    for i in range(n_parts):
        # calculate indices
        i_s = i*part_size       # start index 
        i_e = (i+1)*part_size   # end index
        # create part
        x_t[i_s:i_e], y_t[i_s:i_e] = get_part(
                                        x = x[i_s:i_e], 
                                        y = y[i_s:i_e], 
                                        n_points = min_points
                                    )
        # increase number of points
        min_points += 1
    # shuffling
    x_t, y_t = shuffle_arrays(x_t, y_t)
    return (x_t, y_t)
    

def seedGenerator(x, y, batch_size, shuffle=True):
    minNPointsInTracklet = 3
    # transform input sequences into dataset
    x, y = get_dataset(x, y, minNPointsInTracklet)
    while True:
        # this loop produces batches
        for b in range(len(x) // batch_size):
            batch_xs = x[b*batch_size : (b+1)*batch_size]
            batch_ys = y[b*batch_size : (b+1)*batch_size]
            # additional shuffling
            if shuffle:
                batch_xs, batch_ys = shuffle_arrays(batch_xs, batch_ys)
            yield (batch_xs, batch_ys)


def mkdir(dirname):
    '''Creates directory. If already exists
    removes the existed version
    '''
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
