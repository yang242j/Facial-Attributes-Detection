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
train_img_filenameList = list(df_part[df_part.partition == 0].image_id)
valid_img_filenameList = list(df_part[df_part.partition == 1].image_id)
test_img_filenameList = list(df_part[df_part.partition == 2].image_id)
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
        img_fullpath = config.os.path.sep.join([config.WILD_IMAGES_PATH, filename])
        (h, w, c) = cv2.imread(img_fullpath).shape
        x_1 = float(df.x_1) / w
        y_1 = float(df.y_1) / h
        x_2 = float(df.x_1 + df.width) / w
        y_2 = float(df.y_1 + df.height) / h
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

# Stage_3: Facial landmarks data preparation
def collect_landmarks(filenameList, df_land) -> np.array:
    target_list = []
    for filename in config.progressBar(filenameList, prefix = 'Progress:', suffix = 'Complete', length = 20):
        df = df_land[df_land['image_id'] == filename]
        lefteye_x = float(df.lefteye_x) / config.IMG_WIDTH
        lefteye_y = float(df.lefteye_y) / config.IMG_HEIGHT
        righteye_x = float(df.righteye_x) / config.IMG_WIDTH
        righteye_y = float(df.righteye_y) / config.IMG_HEIGHT
        nose_x = float(df.nose_x) / config.IMG_WIDTH
        nose_y = float(df.nose_y) / config.IMG_HEIGHT
        leftmouth_x = float(df.leftmouth_x) / config.IMG_WIDTH
        leftmouth_y = float(df.leftmouth_y) / config.IMG_HEIGHT
        rightmouth_x = float(df.rightmouth_x) / config.IMG_WIDTH
        rightmouth_y = float(df.rightmouth_y) / config.IMG_HEIGHT
        target_list.append((lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y))
    return np.array(target_list, dtype='float32')

print('[INFO] Collecting Facial Landmarks Array...')
train_landmark = collect_landmarks(train_img_filenameList, df_land)
print('Train Landmark Done: ', train_landmark.shape)
valid_landmark = collect_landmarks(valid_img_filenameList, df_land)
print('Valid Landmark Done: ', valid_landmark.shape)
test_landmark = collect_landmarks(test_img_filenameList, df_land)
print('Test Landmark Done: ', test_landmark.shape)

print('[INFO] Storing Train Image Landmark...', end='')
np.savetxt(config.TRAIN_LANDMARK, train_landmark, fmt='%s')
print('Done')

print('[INFO] Storing Valid Image Landmark...', end='')
np.savetxt(config.VALID_LANDMARK, valid_landmark, fmt='%s')
print('Done')

print('[INFO] Storing Test Image Landmark...', end='')
np.savetxt(config.TEST_LANDMARK, test_landmark, fmt='%s')
print('Done')

# Stage_4: Facial Attributes data preparation
def collect_attr(filenameList, df_attr) -> np.array:
    target_list = []
    for filename in config.progressBar(filenameList, prefix = 'Progress:', suffix = 'Complete', length = 20):
        df = df_attr[df_attr['image_id'] == filename]
        target_list.append(df.values.tolist()[0][1:])
    return np.array(target_list, dtype='int')

df_attr.replace(to_replace=-1, value=0, inplace=True) # convert (-1, 1) to (0, 1) by replacing all -1 to 0
print('[INFO] Collecting Facial Attributes Array...')
train_attr = collect_attr(train_img_filenameList, df_attr)
print('Train Attr Done: ', train_attr.shape)
valid_attr = collect_attr(valid_img_filenameList, df_attr)
print('Valid Attr Done: ', valid_attr.shape)
test_attr = collect_attr(test_img_filenameList, df_attr)
print('Test Attr Done: ', test_attr.shape)
attr_list = df_attr.columns.values.tolist()[1:]
print('Collect Attr Lits Done: ', len(attr_list))

print('[INFO] Storing Train Image Attr...', end='')
np.savetxt(config.TRAIN_ATTR, train_attr, fmt='%s')
print('Done')

print('[INFO] Storing Valid Image Attr...', end='')
np.savetxt(config.VALID_ATTR, valid_attr, fmt='%s')
print('Done')

print('[INFO] Storing Test Image Attr...', end='')
np.savetxt(config.TEST_ATTR, test_attr, fmt='%s')
print('Done')

print('[INFO] Storing Attr List...', end='')
np.savetxt(config.ATTR_LIST, attr_list, fmt='%s')
print('Done')