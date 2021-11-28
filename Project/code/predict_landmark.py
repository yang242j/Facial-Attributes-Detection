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

def landmark_predict_decode(new_h, new_w, model, image) -> None:
    # Make bounding box prediction
    predict = model.predict(image)[0]
    (p_l_eye_x, p_l_eye_y, p_r_eye_x, p_r_eye_y, p_nose_x, p_nose_y, p_l_mouth_x, p_l_mouth_y, p_r_mouth_x, p_r_mouth_y) = predict

    # Decode the prediction corrdinate
    lefteye_x = int(p_l_eye_x * new_w)
    lefteye_y = int(p_l_eye_y * new_h)
    righteye_x = int(p_r_eye_x * new_w) 
    righteye_y = int(p_r_eye_y * new_h) 
    nose_x = int(p_nose_x * new_w) 
    nose_y = int(p_nose_y * new_h)
    leftmouth_x = int(p_l_mouth_x * new_w) 
    leftmouth_y = int(p_l_mouth_y * new_h) 
    rightmouth_x = int(p_r_mouth_x * new_w) 
    rightmouth_y = int(p_r_mouth_y * new_h)

    print(f"Predict Left Eye: ({lefteye_x}, {lefteye_y})")
    print(f"Predict Right Eye: ({righteye_x}, {righteye_y})")
    print(f"Predict Nose: ({nose_x}, {nose_y})")
    print(f"Predict Left Mouth: ({leftmouth_x}, {leftmouth_y})")
    print(f"Predict Right Mouth: ({rightmouth_x}, {rightmouth_y})")

    return (lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

def plot_save_fig(image, img_filename, landmark_predict) -> None:

    # Draw the predicted facial landmarks
    landmark_border_thickness = 5
    circle_radius = 3

    left_eye_color = (255, 0, 255) # Pink
    left_eye = (landmark_predict[0], landmark_predict[1])
    cv2.circle(
        img=image, 
        center=left_eye,
        radius=circle_radius, 
        color=left_eye_color, 
        thickness=landmark_border_thickness
    )

    right_eye_coloe = (255, 128, 0) # Orange
    right_eye = (landmark_predict[2], landmark_predict[3])
    cv2.circle(
        img=image, 
        center=right_eye,
        radius=circle_radius, 
        color=right_eye_coloe, 
        thickness=landmark_border_thickness
    )

    nose_color = (0, 0, 255) # Blue
    nose = (landmark_predict[4], landmark_predict[5])
    cv2.circle(
        img=image, 
        center=nose, 
        radius=circle_radius,
        color=nose_color, 
        thickness=landmark_border_thickness
    )

    left_mouth_color = (255, 255, 0) # Yellow
    left_mouth = (landmark_predict[6], landmark_predict[7])
    cv2.circle(
        img=image, 
        center=left_mouth,
        radius=circle_radius, 
        color=left_mouth_color, 
        thickness=landmark_border_thickness
    )

    right_mouth_color = (128, 0, 255) # Purple
    right_mouth = (landmark_predict[8], landmark_predict[9])
    cv2.circle(
        img=image, 
        center=right_mouth,
        radius=circle_radius, 
        color=right_mouth_color, 
        thickness=landmark_border_thickness
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

    # Load trained facial landmarks model
    print('[INFO] Loading facial-landmarks detecter...', end='')
    landmark_model = load_model(config.LANDMARK_MOEL_PATH)
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

        # Predict landmarks coordinates
        landmark_predict = landmark_predict_decode(new_h, new_w, landmark_model, image_data)
    
        # Plot and save the output image
        plot_save_fig(image, img_filename, landmark_predict)

if __name__ == '__main__':
    main()
    # Activate conda env:
    # source ~/miniforge3/bin/activate
    #
    # Run command: work
    # python predict_landmark.py -I testIMG.txt
    # python predict_landmark.py -I align_testimg.jpg
    # python predict_landmark.py -I ../celeba_dataset/img_align_celeba/img_align_celeba/159902.jpg

    # Run command: NOT work
    # python predict_landmark.py -I 0_face.jpg
    # python predict_landmark.py -I wild_testimg.jpg
    # python predict_landmark.py -I ../celeba_dataset/in_the_wild_celeba/in_the_wild_celeba/159902.jpg
    