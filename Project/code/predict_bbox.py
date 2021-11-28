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
        "-I", 
        "--input", 
        required=True, 
        help="test_image path OR test_image_names.txt path"
    )
    return vars(ap.parse_args())

def get_img_pathlist(inputArg) -> list:
    filetype = mimetypes.guess_type(inputArg)[0]
    img_path = []
    if (filetype == 'text/plain'): #test_img_list.txt
        print("Input .txt file")
        filenames = open(inputArg).read().strip().split('\n')
        for filename in filenames: 
            full_path = config.os.path.sep.join([config.ALIGN_IMAGES_PATH, filename])
            img_path.append(full_path)
    elif(config.os.path.exists(inputArg) and filetype == 'image/jpeg'):#/full_path/test_img.jpg
        print("Input .jpg file with full path, image found")
        img_path.append(inputArg)
    else:#(filetype == 'image/jpeg'): #test_img.jpg
        align_full_path = config.os.path.sep.join([config.ALIGN_IMAGES_PATH, inputArg])
        img_path.append(align_full_path)
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

def bbox_predict_decode(new_h, new_w, model, image) -> None:
    # Make bounding box prediction
    predict = model.predict(image)[0]
    (p_x_1, p_y_1, p_x_2, p_y_2) = predict

    # Decode the prediction corrdinate
    x_1 = int(p_x_1 * new_w)
    y_1 = int(p_y_1 * new_h)
    x_2 = int(p_x_2 * new_w)
    y_2 = int(p_y_2 * new_h)

    print(f"Predict Point_1: ({x_1}, {y_1})")
    print(f"Predict Point_2: ({x_2}, {y_2})")

    return (x_1, y_1, x_2, y_2)

def plot_save_fig(image, img_filename, bbox_predict) -> None:
    
    # Draw the predicted bounding box
    bbox_color = (0, 255, 0) # Green
    bbox_border_thickness = 2
    pt1 = (bbox_predict[0], bbox_predict[1])
    pt2 = (bbox_predict[2], bbox_predict[3])
    cv2.rectangle(
        img=image, 
        pt1=pt1, 
        pt2=pt2, 
        color=bbox_color, 
        thickness=bbox_border_thickness
    )

    # Plot image
    # cv2.imwrite(config.os.path.join(config.BASE_OUTPUT , img_filename), image)
    cv2.imshow(img_filename, image)
    cv2.waitKey(0)

def main() -> None:
    
    # Get arguments input from command
    args = argument_parser()

    print('Input image: ', args['input'])

    imagePaths = get_img_pathlist(args['input'])

    # Load trained bounding box model
    print('[INFO] Loading bounding-box detecter...', end='')
    bbox_model = load_model(config.BBOX_MODEL_PATH)
    print('Done')

    # Loop over the testing images,
    for index, imagePath in enumerate(imagePaths):
        # Load and preprocess input image
        image_data = preprocess_input(imagePath)
        img_filename = str(index) + "_face.jpg"

        # Resize the input image
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=500)
        (new_h, new_w, c) = image.shape

        # Predict the bbox output of the given image
        bbox_predict = bbox_predict_decode(new_h, new_w, bbox_model, image_data)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_predict

        # Extract and save face image out of the input image
        face_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        cv2.imwrite(img_filename, face_image)
    
        # Plot and save the output image
        plot_save_fig(image, img_filename, bbox_predict)

if __name__ == '__main__':
    main()
    # Activate conda env:
    # source ~/miniforge3/bin/activate
    #
    # Run command: work
    # python predict_bbox.py -I testIMG.txt
    # python predict_bbox.py -I align_testimg.jpg
    # python predict_bbox.py -I ../celeba_dataset/img_align_celeba/img_align_celeba/159902.jpg

    # Run command: NOT work
    # python predict_bbox.py -I wild_testimg.jpg
    # python predict_bbox.py -I ../celeba_dataset/in_the_wild_celeba/in_the_wild_celeba/159902.jpg