# IMPORT
import cv2
import config
import numpy as np
import pandas as pd

# Stage_0: Extract csv features
# Load csv files into pandas dataframes
df_attr = pd.read_csv(config.ATTR_PATH)
df_bbox = pd.read_csv(config.BBOX_PATH)
df_part = pd.read_csv(config.PART_PATH)
df_land = pd.read_csv(config.LAND_PATH)

# Stage_1: Split image filename using partition.csv
# Image filename partition
print('[INFO] Collecting Filename Lists...', end='')
train_img_filenameList = list(df_part[df_part.partition == 0].image_id)#[:1000]
valid_img_filenameList = list(df_part[df_part.partition == 1].image_id)#[:100]
test_img_filenameList = list(df_part[df_part.partition == 2].image_id)#[:100]
print('Done')

print('[INFO] Storing Train Image Filenames...', end='')
np.savetxt(config.TRAIN_FILENAMES, train_img_filenameList, fmt='%s')
print('Done')

print('[INFO] Storing Valid Image Filenames...', end='')
np.savetxt(config.VALID_FILENAMES, valid_img_filenameList, fmt='%s')
print('Done')

print('[INFO] Storing Test Image Filenames...', end='')
np.savetxt(config.TEST_FILENAMES, test_img_filenameList, fmt='%s')
print('Done')

# Stage_2: Bounding box data preparation
def collect_bbox(filenameList, df_bbox) -> np.array:
    target_list = []
    for filename in config.progressBar(filenameList, prefix = 'Progress:', suffix = 'Complete', length = 20):
        df = df_bbox[df_bbox['image_id'] == filename]
        (h, w, c) = cv2.imread(config.os.path.sep.join([config.WILD_IMAGES_PATH, filename])).shape
        x_1 = float(df.x_1) / w
        y_1 = float(df.y_1) / h
        x_2 = float(df.x_1 + df.width) / w
        y_2 = float(df.y_1 + df.height) / h
        # print("new bbox: ", x_c, y_c, width, height)
        target_list.append((x_1, y_1, x_2, y_2))

    return np.array(target_list, dtype='float32')

print('[INFO] Collecting Bounding Box Array...')
train_bbox = collect_bbox(train_img_filenameList, df_bbox)
print('Train BBox Done: ', train_bbox.shape)
valid_bbox = collect_bbox(valid_img_filenameList, df_bbox)
print('Valid BBox Done: ', valid_bbox.shape)
test_bbox = collect_bbox(test_img_filenameList, df_bbox)
print('Test BBox Done: ', test_bbox.shape)

print('[INFO] Storing Train Image BBOX...', end='')
np.savetxt(config.TRAIN_BBOX, train_bbox, fmt='%s')
print('Done')

print('[INFO] Storing Valid Image BBox...', end='')
np.savetxt(config.VALID_BBOX, valid_bbox, fmt='%s')
print('Done')

print('[INFO] Storing Test Image BBox...', end='')
np.savetxt(config.TEST_BBOX, test_bbox, fmt='%s')
print('Done')