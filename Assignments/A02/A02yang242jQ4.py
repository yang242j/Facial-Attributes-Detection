# =============================================================================
# #
# #  Created on Wed. Oct 6 2021
# #
# #  ENSE 885AU: A02-Q4
# #
# #  @author: Jingkang Yang <yang242j@uregina.ca>
# #
# #  Execution Command: python A02yang242jQ4.py
# #
# #  Question Declaration:
# #
# #     Consider the l2-regularized logistic regression by adding the given l2 loss term.
# #     -- w and b parameters are from the regression.
# #     -- lamda is the weight for the regularization, lamda = C / n
# #         -- C is some constant in [0.01; 100], to be tuned
# #         -- n is the number of data points
# #     Run the l2-regularized logistic regression on MNIST dataset.
# #     Compare the results with A02yang242jQ3
# #
# =============================================================================
# IMPORTS
import numpy as np
import gzip


class MNIST:

    def __init__(self, IMAGE_train_path, LABEL_train_path, IMAGE_test_path, LABEL_test_path):

        # Extract images and labels from MNIST dataset
        self.IMAGE_training = self.__extract_images(IMAGE_train_path)
        self.LABEL_training = self.__extract_labels(LABEL_train_path)
        self.IMAGE_testing = self.__extract_images(IMAGE_test_path)
        self.LABEL_testing = self.__extract_labels(LABEL_test_path)

        # Normalize the Images pixels from 0-255 to 0-1 for more efficient computation
        self.norm_IMAGE_training = np.true_divide(self.IMAGE_training, 255)
        self.norm_IMAGE_testing = np.true_divide(self.IMAGE_testing, 255)

        # Convert the Labels vector through one-hot-encoder
        self.onehot_LABEL_training = self.__onehot_encoder(self.LABEL_training)
        self.onehot_LABEL_testing = self.__onehot_encoder(self.LABEL_testing)

        # Split training datasets to training and validation dataset
        self.new_IMG_train, self.new_LABEL_train, self.new_IMG_valid, self.new_LABEL_valid = self.__train_validat_split(
            self.norm_IMAGE_training, self.onehot_LABEL_training)

    def __extract_images(self, file_path):
        """
        Define image extraction
        Input: image file_path
        Output: image array
        """
        with gzip.open(file_path, 'rb') as f:
            # magic number, 4 bytes
            magic_number = int(f.read(4).encode('hex'), 16)  # 2051
            # number of images, 4 bytes
            num_images = int(f.read(4).encode('hex'), 16)  # 60000
            # number of rows, 4 bytes
            num_rows = int(f.read(4).encode('hex'), 16)  # 28
            # number of columns, 4 bytes
            num_columns = int(f.read(4).encode('hex'), 16)  # 28
            # image pixels, unsigned bytes each
            # Pixels are organized row-wise.
            # Pixel values are 0 to 255.
            # 0 means background, white. 255 means foreground, black.
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape(
                (num_images, num_rows, num_columns))
            f.close()
        return images

    def __extract_labels(self, file_path):
        """
        Define labels extraction
        Input: labels file_path
        Output: labels array
        """
        with gzip.open(file_path, 'rb') as f:
            # magic number, 4 bytes
            magic_number = int(f.read(4).encode('hex'), 16)  # 2049
            # number of items, 4 bytes
            num_labels = int(f.read(4).encode('hex'), 16)  # 60000
            # labels, unsigned bytes each
            # The labels values are 0 to 9
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            f.close()
        return labels

    def __train_validat_split(self, x_dataset, y_dataset, split_rate=0.25):
        """Split the x_dataset and y_dataset into
        cooresponding training dataset and validation dataset.
        split rate is default to be 0.25
        """
        # Define random starting index and ending index
        starting_index = np.random.choice(range(len(x_dataset)))
        index_range = len(x_dataset) * split_rate
        ending_index = starting_index + index_range

        # Dataset initalization
        x_training_set = np.concatenate(
            (x_dataset[:starting_index], x_dataset[ending_index:]))
        y_training_set = np.concatenate(
            (y_dataset[:starting_index], y_dataset[ending_index:]))
        x_validation_set = x_dataset[starting_index:ending_index]
        y_validation_set = y_dataset[starting_index:ending_index]

        return x_training_set, y_training_set, x_validation_set, y_validation_set

    def __onehot_encoder(self, label_vector):
        """
        One hot encoder
        Input:  label_vector -> vector array of the all the labels
        Output: one hot encoded matrix array of the labels
        """
        # Define a matrix of zeros
        onehot_encode_matrix = np.zeros(shape=(len(label_vector), 10))

        # For each label, update the labeled spots to 1
        for label_index in range(len(label_vector)):
            onehot_encode_matrix[label_index, label_vector[label_index]] = 1

        return np.array(onehot_encode_matrix)


class l2RegularizedLogisticRegression:

    def __init__(self, x_train, y_train, x_valid, y_valid, min_acc):

        self.num_labels = y_train.shape[1]  # 10
        self.num_image_pixel = x_train.shape[1] * x_train.shape[2]  # 28*28=784
        self.num_train_sample = x_train.shape[0]  # 60000(0.75)=45000
        self.min_acc = min_acc

        # Reshape the traing_images to one line vector shape=(?, 28*28)
        self.x_train = np.reshape(x_train, newshape=(
            x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        self.y_train = y_train

        # Reshape the validating_images to one line vector shape=(?, 28*28)
        self.x_valid = np.reshape(x_valid, newshape=(
            x_valid.shape[0], x_valid.shape[1] * x_valid.shape[2]))
        self.y_valid = y_valid

        # Init weights and bias
        self.weights = np.zeros((self.num_labels, self.num_image_pixel))
        self.bias = np.zeros(10)

    def fit_l2(self, eta=0.1, C=0.01):
        """
        Logistic Regression
        Input: x->traing_images, y->traing_labels, learning_rate->eta
        Output: weight after training.
        """
        # Parameter Initialization
        num_iterations = 1
        accuracy = 0
        done = False
        sample_index = 0
        # Training loop
        while (done == False):
            gamma = C / len(self.x_train)
            w_gred, b_gred = self.__gradient(sample_index)
            self.weights -= 2 * eta * w_gred - 2 * eta * gamma * self.weights  # Get new w
            self.bias -= 2 * eta * b_gred - 2 * eta * gamma * self.bias  # Get new w
            # Stoping Criterion: stop if validation accuracy is above 85%
            sample_index += 1
            if sample_index >= len(self.x_train):
                print("Iteration #%d," % num_iterations),
                ce_loss = self.__cross_entropy()
                # print("Cross Entropy Loss: %d," % ce_loss),
                accuracy = self.__validation()
                num_iterations += 1
                sample_index = 0
            done = True if accuracy >= self.min_acc or num_iterations >= 10 else False

    def __gradient(self, index):
        x_data = self.x_train[index]
        y_data = self.y_train[index]
        w = self.weights
        b = self.bias
        # Comput loss and probability
        alpha_vector = np.dot(w, x_data) + b
        softmax_prob = self.__softmax(alpha_vector)
        predict_difference = softmax_prob - y_data
        # N_size = x_data.shape[0]  # 45000
        w_gradient = predict_difference.reshape(10, 1) * x_data
        b_gradient = predict_difference
        # print(w_gradient[0])
        # print(np.mean(np.all(w_gradient[0] == w_gradient[1])))
        return w_gradient, b_gradient

    def __validation(self):
        alpha_vector = np.dot(self.x_valid, self.weights.T) + self.bias
        softmax = self.__softmax(alpha_vector)
        y_predict = np.argmax(softmax, axis=1)
        y_truth = np.argmax(self.y_valid, axis=1)
        accuracy = np.mean(y_predict == y_truth)
        # accuracy = 0.85
        print("Validation accuracy: %s" % "{:.2%}".format(accuracy))
        return accuracy

    def __cross_entropy(self):
        X = self.x_valid
        Y = self.y_valid
        W = self.weights
        B = self.bias
        N = X.shape[0]
        PHI = self.__softmax(np.dot(X, W.T) + B)
        ce_loss = (-1/N) * np.sum(Y * np.log(PHI))
        return ce_loss

    def __softmax(self, alpha_vector):
        exp = np.exp(alpha_vector)
        return exp / np.sum(exp)


def test_training(x_test, y_test, weight, bias):
    x_test = np.reshape(x_test, newshape=(
        x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    alpha_vector = np.dot(x_test, weight.T) + bias
    softmax = np.exp(alpha_vector) / np.sum(np.exp(alpha_vector))
    y_predict = np.argmax(softmax, axis=1)
    y_truth = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_predict == y_truth)
    return accuracy


if __name__ == "__main__":
    # Define datast path
    IMAGE_train_path = "train-images-idx3-ubyte.gz"
    LABEL_train_path = "train-labels-idx1-ubyte.gz"
    IMAGE_test_path = "t10k-images-idx3-ubyte.gz"
    LABEL_test_path = "t10k-labels-idx1-ubyte.gz"

    # Data extraction and manipulation
    print("Colloecting data and Extracting features..."),
    Dataset = MNIST(IMAGE_train_path, LABEL_train_path,
                    IMAGE_test_path, LABEL_test_path)
    print("Done")

    # Perform multiclass logistic represion on MNIST dataset
    print("Start training...")
    Model = l2RegularizedLogisticRegression(
        Dataset.new_IMG_train, Dataset.new_LABEL_train,
        Dataset.new_IMG_valid, Dataset.new_LABEL_valid, min_acc=0.90)
    Model.fit_l2(eta=0.1, C=100)
    print("Done")

    new_weights = Model.weights
    new_bias = Model.bias

    # Test the training weights in through the test dataset
    test_accuracy = test_training(
        Dataset.norm_IMAGE_testing, Dataset.onehot_LABEL_testing, new_weights, new_bias)
    print("Testing accuracy: %s" % ("{:.2%}".format(test_accuracy)))
