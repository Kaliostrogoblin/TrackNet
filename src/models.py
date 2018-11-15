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
from keras.layers import concatenate
# activations
from keras.activations import softplus


class TrackNet:
    """Creates the model of TrackNet - model for track 
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
        self._set_input_shape(input_shape)
        self.__masked_hack = masked_hack
        self._set_level(level)
        self._build_model()
        self._load_pretrained(weights_file)



    def __new__(cls, input_shape=(None, 3), level=2, weights_file=None, masked_hack=False):
        instance = super(TrackNet, cls).__new__(cls)
        instance.__init__(input_shape, level, weights_file, masked_hack)
        return instance.model


    def _set_input_shape(self, input_shape):
        if len(input_shape) != 2:
            input_shape = (None, 3)
            print("Incorrect input shape, it was set to", input_shape)
        self.input_shape = input_shape


    def _set_level(self, level):
        if level < 1 or level > 3:
            level = 2
            print("Incorrect level, it was set to", level)
        self.level = level


    def _build_model(self):
        input_ = Input(self.input_shape)
        # encode each timestep independently skipping zeros strings
        x = TimeDistributed(Masking(mask_value=0.))(input_)
        # timesteps encoder layer
        if self.__masked_hack:
            x = Conv1D(32, 3, padding='same', activation='relu')(input_)
        else:
            x = Conv1D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        # recurrent layers
        x = GRU(32, dropout=0., return_sequences=True)(x)
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
        self.model = Model(inputs=input_, outputs=output, name='TrackNet')


    def _load_pretrained(self, weights_file):
        if weights_file is not None:
            try:
                self.model.load_weights(weights_file)
                print("[INFO] pretrained weights were loaded successfully")
            except Exception as e:
                print(e)
                print("[INFO] pretrained weights weren't loaded")   