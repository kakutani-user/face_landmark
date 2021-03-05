from tensorflow import keras

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from testing.utils import smoothL1

model = keras.models.load_model('./models/facial_landmark_SqueezeNet.h5', custom_objects={'smoothL1': smoothL1})
model.summary()