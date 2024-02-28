import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial

from Inference.Window import predict_windows
from Utils.Preprocess import Normalize
from Utils.Postprocess import predict_adapter
from Model import MainModel

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path", type=str, default="./Checkpoint_weights/*.h5", help="weights.h5 file")
    arg("--image_path", type=str, default="./Images/*.tif", help="4 channel input tif file")
    arg("--save_path", type=str, default="./Predictions/prediction.tif", help="1 channel output tif file")
    arg("--batch_size", type=int, default=4, help="batch size each predict, lowering to reduce memory requirement")
    arg("--strength", type=int, default=3, help="postprocessing parameter, amplify signal for small region")

    return parser.parse_args()


import sys, glob
sys.path.append("./Utils")
sys.path.append("./Model")
from Inference import Window
from Configs import model_get_args


def predict(image_path = "./image.tif", save_path = "./prediction.tif", weight_path = "./checkpoint.weights.h5", 
            batch_size = 4, strength = 3, search_path = True):
    if search_path:
        weight_path = glob.glob(f"{weight_path}", recursive=True)[0]
        image_path = glob.glob(f"{image_path}", recursive=True)[0]

    lows, highs = np.array([ 54., 133.,  48.,  56.]), np.array([ 564., 1120., 1512., 4259.])
    preprocess = partial(Normalize.preprocess, lows = lows, highs = highs)
    input_dim = 4
    predict_dim = 1
    args = model_get_args()
    
    def get_model(weight_path, args):
        model = MainModel.U2Net_dilated(**args)
        model.load_weights(weight_path)

    model = get_model(weight_path, args)
    predictor = partial(predict_adapter, model = model)

    predict_windows(pathTif = image_path, pathSave = save_path, predictor = predictor, preprocess = preprocess,
                    window_size = 512, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "float32", batch_size = 4)

if __name__ == "main":
    main_args = get_main_args()
    predict(**vars(main_args))