from os import sep
from os.path import join, dirname, realpath

PROJECT_ROOT = join(sep, *dirname(realpath(__file__)).split(sep)[: -1])
print(PROJECT_ROOT)
TIMESTAMP = "%Y-%m-%d_%H-%M-%S"

TMP_RESULTS_DIR = join(PROJECT_ROOT, "result", TIMESTAMP)
ARGS_FILE = join(TMP_RESULTS_DIR, "args.yaml")
TRAIN_LOG_FILE = join(TMP_RESULTS_DIR, "log.txt")

DATA_PATH = "../dataset/rubber_data_2021/*/*.tif"
CSV_PATH = "../dataset/rubber_data_2021/func_data_norm.csv"