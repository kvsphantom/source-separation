#Logging format
FORMAT = "[%(filename)s: %(funcName)s] %(message)s]"
FORMAT2 = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"

######## FOR TRAINING PART - RETRAIN_RESNET.PY ##############################
#MODEL_WEIGHT_BASEPATH='/home/kvshenoy/project/code/source-separation/weights'
#MODEL_WEIGHT_BASEPATH='/homedtic/vshenoykadandale/source-separation/fps2/weights'
MODEL_WEIGHT_BASEPATH='/home/venkatesh/project/weights/source-separation'

#PATH_TO_LOG_FILE='/home/kvshenoy/project/code/source-separation/info_file.txt'
#PATH_TO_LOG_FILE='/homedtic/vshenoykadandale/source-separation/fps2/info_file.txt'
PATH_TO_LOG_DIR='/home/venkatesh/project/code/source-separation/logs/'

#PATH_TO_TRAIN_VAL_DIR='/home/kvshenoy/project/code/source-separation/dataset'
#PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/source-separation/source-separation/dataset'
#PATH_TO_TRAIN_VAL_DIR='/homedtic/vshenoykadandale/dataset/juan/splits_fps2'
#PATH_TO_TRAIN_VAL_DIR='/home/kvshenoy/project/dataset/juan/split'
PATH_TO_TRAIN_VAL_DIR='/home/venkatesh/project/dataset/juan/split'

#BATCH_SIZE=16
BATCH_SIZE=32
#BATCH_SIZE=64

#NUM_CLASSES=4
NUM_CLASSES=8



############### FOR DATA SPLIT PART ##########################################
#INPUT_DIR_PATH='/homedtic/vshenoykadandale/dataset/juan/fps2'
PATH_TO_UNCUT_DATASET_FRAMES='/home/venkatesh/project/dataset/juan/fps2'

############### FOR DUO DATA PREP PART ##########################################
TIMESTAMPS='../resources/times.npy'
#PATH_TO_RAW_DATASET='/home/kvshenoy/project/dataset/juan/uncut'
PATH_TO_UNCUT_DATASET='/home/venkatesh/project/dataset/juan/uncut'
#PATH_TO_ORIGINAL_DATASET='/home/kvshenoy/project/dataset/juan/original'
PATH_TO_CUT_DATASET='/home/venkatesh/project/dataset/juan/cut'
#PATH_TO_AV_DUMPS='/home/kvshenoy/project/dataset/juan/dual'
PATH_TO_AV_DUMPS='/home/venkatesh/project/dataset/juan/dual'

############### FOR WEAK LABEL GENERATION PART ##########################################
#PATH_TO_RETRAINED_RESNET_WEIGHTS='/home/kvshenoy/project/code/ResNET_weights/weights/epoch.00-val_loss.0.001-acc.1.000.hdf5'
#PATH_TO_RETRAINED_RESNET_WEIGHTS='/home/venkatesh/project/code/ResNET_weights/weights/epoch.00-val_loss.0.002-acc.1.000.hdf5'
PATH_TO_RETRAINED_RESNET_WEIGHTS='/home/venkatesh/project/weights/source-separation/epoch.00-val_loss.0.002-acc.1.000.hdf5'
BINARIZATION_THRESHOLD=0.3