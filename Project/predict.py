# IMPORTS
import config
import cv2
import imutils
import numpy as np
import mimetypes
import argparse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

def argument_parser() -> dict:
    # Construct command link argument parser and parse the argument
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", 
        "--input", 
        required=True, 
        help="test_image path OR test_image_names.txt path"
    )
    ap.add_argument(
        "-m", 
        "--mode", 
        required=True, 
        help="bbox: bounding-box; OR landmark: facial-landmarks; OR attribute: facial-attributes;"
    )
    return vars(ap.parse_args())

def get_img_pathlist(inputArg) -> list:
    filetype = mimetypes.guess_type(inputArg)[0]
    img_path = []
    if (filetype == 'text/plain'): #test_img_list.txt
        filenames = open(inputArg).read().strip().split('\n')
        for filename in filenames: 
            full_path = config.os.path.sep.join([config.IMAGES_PATH, filename])
            img_path.append(full_path)
    else: #test_img.jpg
        img_path.append(inputArg)
    return img_path

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

def bbox_predict_decode_plot(model, image, img_path) -> None:
    # Make bounding box prediction
    predict = model.predict(image)[0]
    (p_x_1, p_y_1, p_x_2, p_y_2) = predict

    print("p_x_1: ", p_x_1)
    print("p_y_1: ", p_y_1)
    print("p_x_2: ", p_x_2)
    print("p_y_2: ", p_y_2)

    # Load input image, resize it
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    (new_h, new_w, c) = image.shape

    # Decode the prediction corrdinate
    x_1 = int(p_x_1 * new_w)
    y_1 = int(p_y_1 * new_h)
    x_2 = int(p_x_2 * new_w)
    y_2 = int(p_y_2 * new_h)

    print("x1: ", x_1)
    print("y1: ", y_1)
    print("x2: ", x_2)
    print("y2: ", y_2)

    # Draw the predicted bounding box
    border_color = (0, 255, 0)
    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), border_color, c)

    # Plot image
    cv2.imshow('Output', image)
    cv2.waitKey(0)

def main() -> None:
    
    # Get arguments input from command
    args = argument_parser()

    if (args['mode'] != 'bbox' 
    and args['mode'] != 'landmark' 
    and args['mode'] != 'attribute'):
        print("Mode can only be 'bbox' OR 'landmark' OR 'attribute'")
        exit()

    print('Input image: ', args['input'])
    print("Mode select: ", args['mode'])

    imagePaths = get_img_pathlist(args['input'])

    # Load trained bounding box model,
    print('[INFO] Loading bounding-box detecter...'),
    bbox_model = load_model(config.BBOX_MODEL_PATH)
    print('Done')

    # Loop over the testing images,
    for imagePath in imagePaths:
        # Load and preprocess input image
        image = preprocess_input(imagePath)

        # Predict the image ans plot the output
        bbox_predict_decode_plot(bbox_model, image, imagePath)

if __name__ == '__main__':
    main()
    # Run command:
    # python predict.py -i celeba_dataset/in_the_wild_celeba/in_the_wild_celeba/159902.jpg -m bbox