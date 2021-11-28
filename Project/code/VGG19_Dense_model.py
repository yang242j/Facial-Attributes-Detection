# IMPORT
import config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19

class VGG19_Dense:
    def __init__(self, target_size, train_img_filenameList, valid_img_filenameList, test_img_filenameList, image_path):
        self.target_size = target_size
        print('[INFO] Collecting Image Array...', end='')
        train_img_fullpathList = [image_path + filename for filename in train_img_filenameList]
        self.ds_train_img = tf.data.Dataset.from_tensor_slices(train_img_fullpathList)
        valid_img_fullpathList = [image_path + filename for filename in valid_img_filenameList]
        self.ds_valid_img = tf.data.Dataset.from_tensor_slices(valid_img_fullpathList)
        test_img_fullpathList = [image_path + filename for filename in test_img_filenameList]
        self.ds_test_img = tf.data.Dataset.from_tensor_slices(test_img_fullpathList)
        print('Done')

    def __parse_data(self, img_fullpath, targets):
        image = tf.io.read_file(img_fullpath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH))
        return image, targets

    def build_vgg19_dense_model(self, dense_dim, num_output, output_activation) -> Model:
        vgg19_model = VGG19(
            include_top = False, 
            weights = 'imagenet', 
            input_tensor = Input( shape=self.target_size )
        )
        vgg19_model.trainable = False
        
        new_FC_head = Flatten()(vgg19_model.output)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(dense_dim, activation='relu')(new_FC_head)
        new_FC_head = Dense(num_output, activation=output_activation)(new_FC_head)
        
        self.new_model = Model(
            inputs = vgg19_model.input,
            outputs = new_FC_head
        )
        return self.new_model

    def compile_model(self, lossFunc, metrics) -> None:
        self.new_model.compile(
            optimizer = Adam(learning_rate=config.INIT_LR),
            loss = lossFunc,
            metrics = [metrics]
        )

    def fit_model(self, train_target, valid_target, filepath) -> None:
        # Gen train dataset
        ds_train_target = tf.data.Dataset.from_tensor_slices(train_target)
        ds_train = tf.data.Dataset.zip((self.ds_train_img, ds_train_target)).map(self.__parse_data)
        ds_train = ds_train.shuffle(10000).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Gen valid dataset
        ds_valid_target = tf.data.Dataset.from_tensor_slices(valid_target)
        ds_valid = tf.data.Dataset.zip((self.ds_valid_img, ds_valid_target)).map(self.__parse_data)
        ds_valid = ds_valid.shuffle(10000).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Fit model
        earlystopping = EarlyStopping(monitor='val_loss', patience=3)
        checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
        self.H = self.new_model.fit(
            ds_train,
            epochs = config.NUM_EPOCHS,
            validation_data = ds_valid,
            callbacks = [earlystopping, checkpoint]
        )

    def evaluate_model(self, test_target) -> None:
        # Gen test dataset
        ds_test_target = tf.data.Dataset.from_tensor_slices(test_target)
        ds_test = tf.data.Dataset.zip((self.ds_test_img, ds_test_target)).map(self.__parse_data)
        ds_test = ds_test.shuffle(1000).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print('Model Evaluating...')
        self.new_model.evaluate(ds_test, verbose=1)
        self.new_model.summary()

    def save_model(self, fileName, fileType='h5'):
        fileFullName = fileName + '.' + fileType
        filePath = config.os.path.sep.join([config.BASE_OUTPUT, fileFullName])
        self.new_model.save(filePath, save_format=fileType)

    def plot_save_fig(self, train_data, train_data_label, val_data, val_data_label, plt_title, plt_xlabel, plt_ylabel, plt_legend_loc, plt_filename) -> None:
        plt.style.use('ggplot')
        plt.plot(
            self.H.history[train_data], 
            label=train_data_label
        )
        plt.plot(
            self.H.history[val_data], 
            label=val_data_label
        )
        plt.title(plt_title)
        plt.xlabel(plt_xlabel)
        plt.ylabel(plt_ylabel)
        plt.legend(loc=plt_legend_loc)

        # save plot
        plot_path = config.os.path.sep.join([config.BASE_OUTPUT, plt_filename])
        plt.savefig(plot_path)

        # Clear the figure and axes
        plt.clf()
        plt.cla()

# Convert list readed from filepath to n-dim-np-array
def list_to_ndarray(file_path, dtype='float32') -> np.array:
    with open(file_path) as file:
        ndlist = []
        while (line := file.readline().rstrip()):
            ndlist.append(line.split())
    ndarray = np.array(ndlist, dtype=dtype)
    return ndarray