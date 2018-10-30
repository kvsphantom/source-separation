#Logging format
FORMAT = "[%(filename)s: %(funcName)s] %(message)s]"
FORMAT2 = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"

#MODEL_WEIGHT_BASEPATH='/home/kvshenoy/project/code/source-separation/weights'
MODEL_WEIGHT_BASEPATH='/homedtic/vshenoykadandale/source-separation/'

#PATH_TO_LOG_FILE='/home/kvshenoy/project/code/source-separation/info_file.txt'
PATH_TO_LOG_FILE='/homedtic/vshenoykadandale/source-separation/source-separation/info_file.txt'

#PATH_TO_TRAIN_VAL_DIR='/home/kvshenoy/project/code/source-separation/dataset'
#PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/source-separation/source-separation/dataset'
PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/dataset/juan/split'

#BATCH_SIZE=16
BATCH_SIZE=64

#NUM_CLASSES=4
NUM_CLASSES=8
