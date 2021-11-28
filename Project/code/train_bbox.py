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
    test_img_filenameList = open(config.TEST_FILENAMES).read().strip().split('\n')   # 19962
    print('Done')

    # Facial bounding box recognition (upperLeft_x, upperLeft_y, width, height)
    # Using pre-trained VGG19 model
    '''
    Note: 
    The bounding box data is for the original iamge without align and 
    This dataset has wrong bounding boxes datas, low accurarcy is inevitable.
    Original Image Kaggle Dataset: CelebA - Original Wild Images
    '''
    # Load train_bbox, valid_bbox, test_bbox
    train_bbox = modelFunc.list_to_ndarray(config.TRAIN_BBOX)
    valid_bbox = modelFunc.list_to_ndarray(config.VALID_BBOX)
    test_bbox = modelFunc.list_to_ndarray(config.TEST_BBOX)

    # Compile and plot bounding box model
    bbox_model = modelFunc.VGG19_Dense(
        (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3), 
        train_img_filenameList, 
        valid_img_filenameList, 
        test_img_filenameList, 
        config.WILD_IMAGES_PATH
    )
    bbox_model.build_vgg19_dense_model(dense_dim=512, num_output=4, output_activation='sigmoid')
    bbox_model.compile_model(lossFunc='mean_squared_error', metrics='accuracy')
    bbox_model.fit_model(train_target=train_bbox, valid_target=valid_bbox, filepath=config.BBOX_MODEL_PATH)
    bbox_model.evaluate_model(test_bbox)
    bbox_model.plot_save_fig(
        train_data='loss',
        train_data_label='train_mse_loss',
        val_data='val_loss',
        val_data_label='val_mse_loss',
        plt_title='Model train_val_mse_loss',
        plt_xlabel='Epoch',
        plt_ylabel='MSE Loss',
        plt_legend_loc='upper right',
        plt_filename='landmark_train_val_mse_loss.png'
    )
    bbox_model.plot_save_fig(
        train_data='accuracy',
        train_data_label='train_acc',
        val_data='val_accuracy',
        val_data_label='val_acc',
        plt_title='Model train_val_acc',
        plt_xlabel='Epoch',
        plt_ylabel='Accuracy',
        plt_legend_loc='upper left',
        plt_filename='landmark_train_val_acc.png'
    )

    # Test image bounding box prediction
    test_image = load_img(
        path = '../celeba_dataset/img_align_celeba/img_align_celeba/000001.jpg', 
        color_mode = 'rgb', 
        target_size = (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3)
    )
    test_image = img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    (p_x_1, p_y_1, p_x_2, p_y_2) = bbox_model.new_model.predict(test_image)[0]
    print(f"Predict Point_1: ({p_x_1}, {p_y_1})")
    print(f"Predict Point_2: ({p_x_2}, {p_y_2})")


if __name__ == '__main__':
    main()