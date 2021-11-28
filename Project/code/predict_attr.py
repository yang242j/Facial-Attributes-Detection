# IMPORTS
import config
import numpy as np
import pandas as pd
import mimetypes
import argparse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

def argument_parser() -> dict:
    # Construct command link argument parser and parse the argument
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-I", 
        "--input", 
        required=True, 
        help="test_image path OR test_image_names.txt path"
    )
    return vars(ap.parse_args())

def get_img_pathlist(inputArg) -> list:
    filetype = mimetypes.guess_type(inputArg)[0]
    img_path = []
    true_value_list = []
    df_attr = pd.read_csv(config.ATTR_PATH)
    df_attr.replace(to_replace=-1, value=0, inplace=True)
    if (filetype == 'text/plain'): #test_img_list.txt
        print("Input .txt file")
        filenames = open(inputArg).read().strip().split('\n')
        for filename in filenames: 
            full_path = config.os.path.sep.join([config.ALIGN_IMAGES_PATH, filename])
            img_path.append(full_path)
            img_df = df_attr[df_attr['image_id'] == filename]
            true_value_list.append(img_df.values.tolist()[0][1:])
    elif(config.os.path.exists(inputArg) and filetype == 'image/jpeg'):#/full_path/test_img.jpg
        print("Input .jpg file with full path, image found")
        img_path.append(inputArg)
    else:# test_img.jpg
        align_full_path = config.os.path.sep.join([config.ALIGN_IMAGES_PATH, inputArg])
        img_path.append(align_full_path)
        img_df = df_attr[df_attr['image_id'] == inputArg]
        true_value_list.append(img_df.values.tolist()[0][1:])
    return img_path, true_value_list

def preprocess_input(img_path):
    # Load input image in Keras format and preprocess and normalize the image.
    image = load_img(
        path = img_path, 
        color_mode = 'rgb', 
        target_size = (config.TAR_IMG_HEIGHT, config.TAR_IMG_WIDTH, 3)
    )
    image = img_to_array(image) / 255.
    image = np.expand_dims(image, axis=0)
    return image

def main() -> None:
    
    # Get arguments input from command
    args = argument_parser()

    print('Input image: ', args['input'])

    imagePaths, true_value_list = get_img_pathlist(args['input'])

    # Load trained bounding box model, and facial landmarks model
    print('[INFO] Loading facial-attributes detecter...', end='')
    attr_model = load_model(config.ATTR_MODEL_PATH)
    print('Done')

    # Load the attributes list
    attribut_list = open(config.ATTR_LIST).read().strip().split('\n')
    
    # Loop over the testing images,
    for index, imagePath in enumerate(imagePaths):
        # Load and preprocess input image
        image_data = preprocess_input(imagePath)

        # Predict probabilities for each facial attributes
        attr_predict = attr_model.predict(image_data)[0]
        for i, pred_prob in enumerate(attr_predict):
            if true_value_list:
                # Load true value of each input image, if exist
                true_attr = true_value_list[index]
                print(attribut_list[i], '->', ' Predict: ', "{:.0%}".format(pred_prob), ' Actual: ', true_attr[i])
            else:
                print(attribut_list[i], '->', ' Predict: ', "{:.0%}".format(pred_prob))


if __name__ == '__main__':
    main()
    # Activate conda env:
    # source ~/miniforge3/bin/activate
    #
    # Run command: work
    # python predict_attr.py -I testIMG.txt
    # python predict_attr.py -I align_testimg.jpg
    # python predict_attr.py -I ../celeba_dataset/img_align_celeba/img_align_celeba/159902.jpg

    # Run command: NOT work
    # python predict_attr.py -I 0_face.jpg
    # python predict_attr.py -I wild_testimg.jpg
    # python predict_attr.py -I ../celeba_dataset/in_the_wild_celeba/in_the_wild_celeba/159902.jpg