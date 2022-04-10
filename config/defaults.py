from os.path import join

from yacs.config import CfgNode

from config.const import PROJECT_ROOT


_C = CfgNode()
_C.SEED = 42


# train
_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.GPUS = 2
_C.TRAIN.LR = 1e-3
_C.TRAIN.MODEL_TYPE = 'vgg'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER_TYPE = 'SGD'
_C.TRAIN.KFOLD = 5
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.EARLY_STOP = 100

# 
_C.DRAW_PROCESS = True