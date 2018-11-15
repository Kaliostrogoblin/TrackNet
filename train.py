import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-g", "--gpus", type=int, default=0,
    help = "# of GPUs to use for training")
ap.add_argument("--n_epochs", type=int, default=50,
    help="Number of epochs of TreeNet")
ap.add_argument("--batch_size", type=int, default=128,
    help="Number of samples per gradient update")
ap.add_argument("--inference", type=bool, default=False,
    help="If inference=True, save the inference speed")
ap.add_argument("--data_dir", type=str, default="data",
    help="Directory with the data for training")
ap.add_argument("--masked_hack", type=bool, default=False,
    help="Disable masking layer")
args = vars(ap.parse_args())

G = args["gpus"]                # number of GPUs
batch_size = args["batch_size"] # samples per gradient step
n_epochs = args["n_epochs"]     # training epochs
data_dir = args["data_dir"]     # training epochs
masked_hack = args["masked_hack"] # hack

# import modules
if G > 1:
    from keras.utils import multi_gpu_model

# data generator
from src.data_utils import seedGenerator
from src.data_utils import read_data
from src.data_utils import get_dataset
# model
from src.models import TrackNet
# callbacks
from src.callbacks import ModelCheckpoint
from src.callbacks import TimingCallback
from src.callbacks import Metrics
# losses
from src.losses import custom_loss
# metrics
from src.metrics import accuracy
from src.metrics import point_in_area
from src.metrics import circle_area
# other imports
from keras.optimizers import Adam
import os


if __name__ == '__main__':
    print('[INFO] getting data...')
    (x_train, y_train), (x_test, y_test) = read_data(data_dir)
    # create generators
    steps_per_epoch = len(x_train) // batch_size
    #validation_steps = len(x_test) // batch_size
    # create generator instances
    train_gen = seedGenerator(x_train, y_train, batch_size)
    x_test, y_test = get_dataset(x_test, y_test, 3)

    if G == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        print("[INFO] training with CPU...")
        tracknet = TrackNet(masked_hack = masked_hack)
    elif G == 1:
        print("[INFO] training with 1 GPU...")
        tracknet = TrackNet(masked_hack = masked_hack)
    else:
        batch_size = batch_size * G # correct the batch size
        print("[INFO] training with {} GPUs...".format(G))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            tracknet = TrackNet(masked_hack = masked_hack)
        # make the model parallel
        tracknet = multi_gpu_model(tracknet, gpus=G)

    # compile model with loss, metrics and opt
    print("[INFO] compiling model...")
    tracknet.compile(
        loss=custom_loss(lambda1=0.5, 
                         lambda2=0.35, 
                         lambda3=0.15, 
                         focal_alpha=0.95, 
                         focal_gamma=2),
        optimizer=Adam(),
        metrics=[accuracy, point_in_area, circle_area]
    )

    # train the network
    print("[INFO] training network...")
    history = tracknet.fit_generator(
        generator=train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=(x_test, y_test),
        validation_steps=1,
        callbacks = [Metrics()]
    )