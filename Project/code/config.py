# IMPORT
import os

# Define the file path
DATASET_PATH = '../celeba_dataset'
ATTR_PATH = os.path.sep.join([DATASET_PATH, 'list_attr_celeba.csv'])
BBOX_PATH = os.path.sep.join([DATASET_PATH, 'list_bbox_celeba.csv'])
PART_PATH = os.path.sep.join([DATASET_PATH, 'list_eval_partition.csv'])
LAND_PATH = os.path.sep.join([DATASET_PATH, 'list_landmarks_align_celeba.csv'])
ALIGN_IMAGES_PATH = os.path.sep.join([DATASET_PATH, 'img_align_celeba/img_align_celeba/'])
WILD_IMAGES_PATH = os.path.sep.join([DATASET_PATH, 'in_the_wild_celeba/in_the_wild_celeba/'])

# Define the output directory
BASE_OUTPUT = "output"
BBOX_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'bbox_model.h5'])
LANDMARK_MOEL_PATH = os.path.sep.join([BASE_OUTPUT, 'landmark_model.h5'])
ATTR_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'attr_model.h5'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])

# Define the prepared directory
READY_DATA = "ready_data" 
TRAIN_FILENAMES = os.path.sep.join([READY_DATA, 'train_img_filenameList.txt'])
VALID_FILENAMES = os.path.sep.join([READY_DATA, 'valid_img_filenameList.txt'])
TEST_FILENAMES = os.path.sep.join([READY_DATA, 'test_img_filenameList.txt'])
TRAIN_BBOX = os.path.sep.join([READY_DATA, 'train_img_bbox.txt'])
VALID_BBOX = os.path.sep.join([READY_DATA, 'valid_img_bbox.txt'])
TEST_BBOX = os.path.sep.join([READY_DATA, 'test_img_bbox.txt'])
TRAIN_LANDMARK = os.path.sep.join([READY_DATA, 'train_img_landmark.txt'])
VALID_LANDMARK = os.path.sep.join([READY_DATA, 'valid_img_landmark.txt'])
TEST_LANDMARK = os.path.sep.join([READY_DATA, 'test_img_landmark.txt'])
TRAIN_ATTR = os.path.sep.join([READY_DATA, 'train_img_attr.txt'])
VALID_ATTR = os.path.sep.join([READY_DATA, 'valid_img_attr.txt'])
TEST_ATTR = os.path.sep.join([READY_DATA, 'test_img_attr.txt'])
ATTR_LIST = os.path.sep.join([READY_DATA, 'attr_list.txt'])

# Define original image dimension
IMG_WIDTH = 178.0
IMG_HEIGHT = 218.0

# Define traing and predicting target_image_dimension
TAR_IMG_WIDTH = 224
TAR_IMG_HEIGHT = 224

# Define deep learning hyperparameters
INIT_LR = 0.00001
NUM_EPOCHS = 100
BATCH_SIZE = 32

# Supress the tensorflow warning messages
# by adjusting the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:
#       0 = all message are logged (default behavior)
#       1 = INFO messages are not printed
#       2 = INFO and WARNING messages are not printed
#       3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Print iterations progress
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @Author of this Function: Greenstick from stackoverflow
    @URL: https://stackoverflow.com/a/34325723
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()