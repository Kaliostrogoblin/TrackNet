# model
from keras.models import Model
# layers
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import concatenate
# activations
from keras.activations import softplus
from .track_net_v1 import TrackNet

class TrackNetV2(TrackNet):
    """Creates the model of TrackNet version 2 with Conv1D instead of GRU - model for track 
    reconstruction in HEP

    # Arguments
        input_shape     : tuple, optional [default=(None, 3)]
                            input tensor [N, M, F]
                             N = None - batch size, 
                             M = None - sequence length, 
                             F = 3 - x, y, z coords 
        level           : int, optional [default = 2]
                            level 1 - for 2 points: target + any of station0
                            level 2 - for range(3, (n-1)), n - no. stations
                            level 3 - for n points 
        weights_file    : string, path to the h5py file with weights

    # Returns 
        keras Model
    """
    def __init__(self, input_shape, level, weights_file, masked_hack):
        super(TrackNetV2, self).__init__(input_shape, level, weights_file, masked_hack)

    def __new__(cls, input_shape=(None, 3), level=2, weights_file=None, masked_hack=False):
        return super(TrackNetV2, cls).__new__(cls, input_shape, level, weights_file, masked_hack)

    def _build_model(self):
        input_ = Input(self.input_shape)
        # encode each timestep independently skipping zeros strings
        x = TimeDistributed(Masking(mask_value=0.))(input_)
        # timesteps encoder layer
        if self._masked_hack:
            x = Conv1D(32, 3, padding='same', activation='relu')(input_)
        else:
            x = Conv1D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        # conv layers
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(16, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = GRU(16, dropout=0.)(x)
        # outputs list
        outputs = []
        # we can't compute probability of track/ghost for 2 points
        if self.level > 1: 
            # probability: track or not
            prob = Dense(1, activation='sigmoid')(x)
            outputs.append(prob)
        # level 3 - only prediction track or not
        if self.level < 3:
            # x and y coords - centre of observing
            # area on the next station
            xy_coords = Dense(2, activation='linear')(x)
            # ellipse radii
            r1_r2 = Dense(2, activation=softplus)(x)
            outputs.extend([xy_coords, r1_r2])

        if len(outputs) > 1:
            # merge everything to one output
            output = concatenate(outputs)
        else:
            output = outputs[0]
        # create model
        self.model = Model(inputs=input_, outputs=output, name='TrackNetV2')
        print("[DEBUG] model:")
        self.model.summary()