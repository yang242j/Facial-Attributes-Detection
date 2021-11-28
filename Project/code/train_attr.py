# IMPORTS
import config
import VGG19_Dense_model as modelFunc
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def main():

    # Split image filename using partition.csv
    # Image filename partition
    print('[INFO] Collecting Filename Lists...', end='')
    train_img_filenameList = open(config.TRAIN_FILENAMES).read().strip().split('\n') # 162770
    valid_img_filenameList = open(config.VALID_FILENAMES).read().strip().split('\n') # 19867
    test_img_filenameList = open(config.TEST_FILENAMES).read().strip().split('\n') # 19962
    print('Done')

    # Facial bounding box recognition (upperLeft_x, upperLeft_y, width, height)
    # Using pre-trained VGG19 model
    '''
    Note: 
    The bounding box data is for the original iamge without align and 
    This dataset has wrong bounding boxes datas, low accurarcy is inevitable.
    Original Image Kaggle Dataset: CelebA - Original Wild Images
    '''
    # Load train_attr, valid_attr, test_attr
    # ATTR_INDEX = [0, 1, 2]
    train_attr = modelFunc.list_to_ndarray(config.TRAIN_ATTR, dtype='int')#[:1000, ATTR_INDEX]
    valid_attr = modelFunc.list_to_ndarray(config.VALID_ATTR, dtype='int')#[:100, ATTR_INDEX]
    test_attr = modelFunc.list_to_ndarray(config.TEST_ATTR, dtype='int')#[:100, ATTR_INDEX]

    # Compile and plot bounding box model
    attr_model = modelFunc.VGG19_Dense(
        (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3), 
        train_img_filenameList, 
        valid_img_filenameList, 
        test_img_filenameList, 
        config.ALIGN_IMAGES_PATH
    )
    attr_model.build_vgg19_dense_model(dense_dim=512, num_output=40, output_activation='sigmoid')
    attr_model.compile_model(lossFunc='binary_crossentropy', metrics='binary_accuracy')
    attr_model.fit_model(train_target=train_attr, valid_target=valid_attr, filepath=config.ATTR_MODEL_PATH)
    attr_model.evaluate_model(test_attr)
    attr_model.plot_save_fig(
        train_data='loss',
        train_data_label='train_loss',
        val_data='val_loss',
        val_data_label='val_loss',
        plt_title='Model train_val_bce_loss',
        plt_xlabel='Epoch',
        plt_ylabel='BCE Loss',
        plt_legend_loc='upper right',
        plt_filename='attribute_train_val_bce_loss.png'
    )
    attr_model.plot_save_fig(
        train_data='binary_accuracy',
        train_data_label='train_bi_acc',
        val_data='val_binary_accuracy',
        val_data_label='val_bi_acc',
        plt_title='Model train_val_bi_acc',
        plt_xlabel='Epoch',
        plt_ylabel='Binary Accuracy',
        plt_legend_loc='upper left',
        plt_filename='attribute_train_val_bi_acc.png'
    )

    # Test image bounding box prediction
    test_image = load_img(
        path = '../celeba_dataset/img_align_celeba/img_align_celeba/000001.jpg', 
        color_mode = 'rgb', 
        target_size = (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3)
    )
    test_image = img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    true_value = train_attr[0]
    pred_value = attr_model.new_model.predict(test_image)[0]
    attr_list = open(config.ATTR_LIST).read().strip().split('\n')

    for i, predict in enumerate(pred_value):
        print(attr_list[i], '->', ' Predict: ', "{:.0%}".format(predict), ' Actual: ', true_value[i])


if __name__ == '__main__':
    main()