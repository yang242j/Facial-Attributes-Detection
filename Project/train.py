# IMPORTS
import config
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def list_to_ndarray(one_d_delimiter, two_d_delimiter, file_path) -> np.array:
    file = open(file_path).read().strip()
    data_list = file.split(one_d_delimiter)
    ndarray = np.array([
        data.split(two_d_delimiter) for data in data_list
    ], dtype='float32')
    return ndarray

class VGG19_Dense:
    def __init__(self, target_size, train_img_filenameList, valid_img_filenameList, test_img_filenameList):
        self.target_size = target_size
        print('[INFO] Collecting Image Array...'),
        # self.train_img = self.__collect_image(train_img_filenameList)
        train_img_fullpathList = [config.WILD_IMAGES_PATH + filename for filename in train_img_filenameList]
        self.ds_train_img = tf.data.Dataset.from_tensor_slices(train_img_fullpathList)
        valid_img_fullpathList = [config.WILD_IMAGES_PATH + filename for filename in valid_img_filenameList]
        self.ds_valid_img = tf.data.Dataset.from_tensor_slices(valid_img_fullpathList)
        test_img_fullpathList = [config.WILD_IMAGES_PATH + filename for filename in test_img_filenameList]
        self.ds_test_img = tf.data.Dataset.from_tensor_slices(test_img_fullpathList)

    def __collect_image(self, filenameList) -> np.array:
        image_array = np.array([
            np.array(
                img_to_array(load_img(config.os.path.sep.join([config.WILD_IMAGES_PATH, filename]), target_size=self.target_size)),
                dtype='float32'
            ) / 255.0 for filename in config.progressBar(filenameList, prefix = 'Progress:', suffix = 'Complete', length = 20)
        ])
        return image_array
    
    def __parse_data(self, x, y):
        image = tf.io.read_file(x)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = image/255.
        image = tf.image.resize(image, (224, 224))
        return image,y

    def build_vgg19_dense_model(self, num_output, dense_dim=500) -> Model:
        vgg19_model = VGG19(
            include_top = False, 
            weights = 'imagenet', 
            input_tensor = Input( shape=self.target_size )
        )
        vgg19_model.trainable = False
        
        new_FC_head = Flatten()(vgg19_model.output)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(num_output, activation='sigmoid')(new_FC_head)
        
        self.new_model = Model(
            inputs = vgg19_model.input,
            outputs = new_FC_head
        )
        return self.new_model

    def compile_model(self) -> None:
        self.new_model.compile(
            optimizer = Adam(learning_rate=config.INIT_LR),
            loss = 'mse',
            metrics = ['accuracy']
        )

    def fit_model(self, train_target, valid_target) -> None:
        # Gen train dataset
        ds_train_target = tf.data.Dataset.from_tensor_slices(train_target)
        ds_train = tf.data.Dataset.zip((self.ds_train_img, ds_train_target)).map(self.__parse_data)
        ds_train = ds_train.shuffle(10000).batch(config.BATCH_SIZE)

        # Gen valid dataset
        ds_valid_target = tf.data.Dataset.from_tensor_slices(valid_target)
        ds_valid = tf.data.Dataset.zip((self.ds_valid_img, ds_valid_target)).map(self.__parse_data)
        ds_valid = ds_valid.shuffle(10000).batch(config.BATCH_SIZE)

        # Fit model
        # steps_per_epoch = len(list(self.ds_train_img)) // config.BATCH_SIZE
        myCallback = EarlyStopping(monitor='val_loss', patience=3)
        self.H = self.new_model.fit(
            ds_train,
            epochs = config.NUM_EPOCHS,
            # steps_per_epoch = steps_per_epoch,
            validation_data = ds_valid,
            callbacks = [myCallback]
        )

    def evaluate_model(self, test_target) -> None:
        # Gen test dataset
        ds_test_target = tf.data.Dataset.from_tensor_slices(test_target)
        ds_test = tf.data.Dataset.zip((self.ds_test_img, ds_test_target)).map(self.__parse_data)
        ds_test = ds_test.shuffle(10000).batch(config.BATCH_SIZE)

        print('Model Evaluating...')
        self.new_model.evaluate(ds_test, verbose=2)
        self.new_model.summary()

    def save_model(self, fileName='vgg19_dense', fileType='h5'):
        fileFullName = fileName + '.' + fileType
        filePath = config.os.path.sep.join([config.BASE_OUTPUT, fileFullName])
        self.new_model.save(filePath, save_format=fileType)

    def plot_save_fig(self, plot_stype='ggplot', plot_filename='plot.png') -> None:
        # plot the model training history
        plt.style.use(plot_stype)
        plt.figure()
        plt.plot(
            self.H.history['loss'], 
            label='train_loss'
        )
        plt.plot(
            self.H.history['val_loss'], 
            label='val_loss'
        )
        plt.title('Bounding Box Detection Loss')
        plt.xlabel('Epoch #')
        plt.ylabel('MSE Loss')
        plt.legend(
            loc='upper right'
        )
        plot_path = config.os.path.sep.join([config.BASE_OUTPUT, plot_filename])
        plt.savefig(plot_path)

def main():

    # Stage_1: Split image filename using partition.csv
    # Image filename partition
    print('[INFO] Collecting Filename Lists...', end='')
    train_img_filenameList = open(config.TRAIN_FILENAMES).read().strip().split('\n')#[:1000]
    # (162770, 112, 112, 3)
    valid_img_filenameList = open(config.VALID_FILENAMES).read().strip().split('\n')#[:100]
    # (19867, 112, 112, 3)
    test_img_filenameList = open(config.TEST_FILENAMES).read().strip().split('\n')#[:100]
    # (19962, 112, 112, 3)
    print('Done')

    # Stage_2: Facial bounding box recognition (upperLeft_x, upperLeft_y, width, height)
    # Using pre-trained VGG19 model
    '''
    Note: 
    The bounding box data is for the original iamge without align and 
    This dataset has wrong bounding boxes datas, low accurarcy is inevitable.
    e.g. image original shape is 218 in height and 178 in width, but the bbox.csv has a lot numbers greater than 218.
    Conclution: The training in this part has been suspended.
    Original Image Kaggle Dataset: CelebA - Original Wild Images
    '''
    # Load train_bbox, valid_bbox, test_bbox
    train_bbox = list_to_ndarray('\n', ' ', config.TRAIN_BBOX)#[:1000]
    valid_bbox = list_to_ndarray('\n', ' ', config.VALID_BBOX)#[:100]
    test_bbox = list_to_ndarray('\n', ' ', config.TEST_BBOX)#[:100]

    # Compile and plot bounding box model
    bbox_model = VGG19_Dense((config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3), train_img_filenameList, valid_img_filenameList, test_img_filenameList)
    bbox_model.build_vgg19_dense_model(num_output=4, dense_dim=500)
    bbox_model.compile_model()
    bbox_model.fit_model(train_bbox, valid_bbox)
    bbox_model.evaluate_model(test_bbox)
    bbox_model.save_model(fileName='bbox_model', fileType='h5')
    bbox_model.plot_save_fig(plot_filename='bbox_train_valid.png')

    # Test image bounding box prediction
    test_image = load_img(
        path = 'celeba_dataset/in_the_wild_celeba/in_the_wild_celeba/000001.jpg', 
        color_mode = 'rgb', 
        target_size = (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3)
    )
    test_image = img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    (p_x_1, p_y_1, p_x_2, p_y_2) = bbox_model.new_model.predict(test_image)[0]
    print("p_x_1: ", p_x_1)
    print("p_y_1: ", p_y_1)
    print("p_x_2: ", p_x_2)
    print("p_y_2: ", p_y_2)

    # Stage_3: Facial area detection, eye location, nose location, left-right mouth location


if __name__ == '__main__':
    main()
