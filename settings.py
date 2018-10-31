#Logging format
FORMAT = "[%(filename)s: %(funcName)s] %(message)s]"
FORMAT2 = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"

######## FOR TRAINING PART - RETRAIN_RESNET.PY ##############################
#MODEL_WEIGHT_BASEPATH='/home/kvshenoy/project/code/source-separation/weights'
MODEL_WEIGHT_BASEPATH='/homedtic/vshenoykadandale/source-separation/fps2/weights'

#PATH_TO_LOG_FILE='/home/kvshenoy/project/code/source-separation/info_file.txt'
PATH_TO_LOG_FILE='/homedtic/vshenoykadandale/source-separation/fps2/info_file.txt'

#PATH_TO_TRAIN_VAL_DIR='/home/kvshenoy/project/code/source-separation/dataset'
#PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/source-separation/source-separation/dataset'
PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/dataset/juan/splits_fps2'

#BATCH_SIZE=16
BATCH_SIZE=64

#NUM_CLASSES=4
NUM_CLASSES=8



############### FOR DATA SPLIT PART ##########################################
INPUT_DIR_PATH='/homedtic/vshenoykadandale/dataset/juan/fps2'
