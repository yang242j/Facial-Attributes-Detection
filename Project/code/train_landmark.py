# IMPORTS
import config
import VGG19_Dense_model as modelFunc
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def main():

    # Stage_1: Split image filename using partition.csv
    # Image filename partition
    print('[INFO] Collecting Filename Lists...', end='')
    train_img_filenameList = open(config.TRAIN_FILENAMES).read().strip().split('\n')#[:1000] # 162770
    valid_img_filenameList = open(config.VALID_FILENAMES).read().strip().split('\n')#[:100] # 19867
    test_img_filenameList = open(config.TEST_FILENAMES).read().strip().split('\n')#[:100] # 19962
    print('Done')

    # Facial area detection, eye location, nose location, left-right mouth location
    # Using pre-trained VGG19 model
    '''
    Note: 
    The facial landmarks data is based on the aligned image not the original wild image 
    This dataset has wrong bounding boxes datas, low accurarcy is inevitable.
    Aligned Image Kaggle Dataset: CelebFaces Attributes (CelebA) Dataset
    '''
    # Load train_landmark, valid_landmark, test_landmark
    # LANDMARK_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_landmark = modelFunc.list_to_ndarray(config.TRAIN_LANDMARK)#[:1000, LANDMARK_INDEX]
    valid_landmark = modelFunc.list_to_ndarray(config.VALID_LANDMARK)#[:100, LANDMARK_INDEX]
    test_landmark = modelFunc.list_to_ndarray(config.TEST_LANDMARK)#[:100, LANDMARK_INDEX]
    
    # Compile and plot landmark model
    landmark_model = modelFunc.VGG19_Dense(
        (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3), 
        train_img_filenameList, 
        valid_img_filenameList, 
        test_img_filenameList, 
        config.ALIGN_IMAGES_PATH
    )
    landmark_model.build_vgg19_dense_model(dense_dim=4096, num_output=10, output_activation='sigmoid')
    landmark_model.compile_model(lossFunc='mean_squared_error', metrics=['accuracy'])
    landmark_model.fit_model(train_target=train_landmark, valid_target=valid_landmark, filepath=config.LANDMARK_MOEL_PATH)
    landmark_model.evaluate_model(test_landmark)
    landmark_model.plot_save_fig(
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
    landmark_model.plot_save_fig(
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

    # Test image landmark prediction
    test_image = load_img(
        path = '../celeba_dataset/img_align_celeba/img_align_celeba/000001.jpg', 
        color_mode = 'rgb', 
        target_size = (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3)
    )
    test_image = img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    
    # true_value = train_landmark[0]
    # pred_value = landmark_model.new_model.predict(test_image)[0]
    # for i, predict in enumerate(pred_value):
    #     diff = predict-true_value[i]
    #     print('Predict: ', predict, ', Actual: ', true_value[i], ' -> ', diff, ' -> ', "{:.3%}".format(diff))

    (p_l_eye_x, p_l_eye_y, p_r_eye_x, p_r_eye_y, p_nose_x, p_nose_y, p_l_mouth_x, p_l_mouth_y, p_r_mouth_x, p_r_mouth_y) = landmark_model.new_model.predict(test_image)[0]
    print(f"Predict Left Eye: ({p_l_eye_x}, {p_l_eye_y})")
    print(f"Predict Right Eye: ({p_r_eye_x}, {p_r_eye_y})")
    print(f"Predict Nose: ({p_nose_x}, {p_nose_y})")
    print(f"Predict Left Mouth: ({p_l_mouth_x}, {p_l_mouth_y})")
    print(f"Predict Right Mouth: ({p_r_mouth_x}, {p_r_mouth_y})")


if __name__ == '__main__':
    main()