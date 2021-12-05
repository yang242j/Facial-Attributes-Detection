# =============================================================================
# #
# #  Created on Thu. Nov. 4th 2021
# #
# #  ENSE 885AU: A04-Q2
# #
# #  @author: Jingkang Yang <yang242j@uregina.ca>
# #
# #  Execution Command: python A04yang242jQ2.py
# #
# #  Question Declaration:
# #
# #     Based on A03Q4, add Convolutional layers
# #     Compare with the original FNN
# #
# =============================================================================
# IMPORTS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

NUM_OF_EPOCHS = 250
BATCH_SIZE = 128
VALIDATION_SPLIT_RATE = 0.1
NUM_OF_LABELS = 10
HIDDEN_LAYER_DIMENSION = 500


class Neural_Network:

    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def build_FFNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28)),
            Flatten(),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def build_l2norm_FFNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28)),
            Flatten(),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def build_dropout_FFNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28)),
            Flatten(),
            Dropout(0.2),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dropout(0.5),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dropout(0.5),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def build_CNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def build_l2norm_CNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def build_dropout_CNN(self) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.2),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dropout(0.5),
            Dense(HIDDEN_LAYER_DIMENSION, activation='relu'),
            Dropout(0.5),
            Dense(NUM_OF_LABELS, activation='softmax')
        ])

    def compile(self) -> None:
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self) -> None:
        batch_end_loss = list()
        class MyCallback(Callback):
            def on_train_batch_end(self, batch, logs=None):
                batch_end_loss.append(logs['loss'])

        self.HistoryObj = self.model.fit(
            self.train_x,
            self.train_y,
            batch_size=BATCH_SIZE,
            epochs=NUM_OF_EPOCHS,
            validation_split=VALIDATION_SPLIT_RATE,
            callbacks=[MyCallback()]
        )
        self.batch_end_loss = batch_end_loss

    def evaluate(self) -> None:
        print("Model Evaluating:")
        self.model.evaluate(self.test_x,  self.test_y, verbose=2)
        self.model.summary()


def plot_diagram(plot_Dict, nrows, ncols):

    # Define plot figure size, 10_inch_width by 5_inch_height
    plt.figure(figsize=(15, 20))

    # Plot the cross entropy loss on the batches
    for dict_index, (plot_title, plot_data_list) in enumerate(plot_Dict.items()):
        plot_index = dict_index+1
        plot_data_array = np.array(plot_data_list)
        plt.subplot(nrows, ncols, plot_index)
        [plt.plot(lineData) for lineData in plot_data_array]
        plt.title(plot_title)
        if (plot_index == 1):
            plt.xlabel('Batches')
            plt.ylabel('Cross_Entropy_Loss')
            plt.legend(['3_layer FFNN', '3_layer CNN'], loc='upper right')
        elif (plot_index == 2):
            plt.xlabel('Validation Data')
            plt.ylabel('Classification_Error_Rate')
            plt.legend(['3_layer FFNN', '3_layer CNN'], loc='upper right')
        elif (plot_index == 3):
            plt.xlabel('Batches')
            plt.ylabel('Cross_Entropy_Loss')
            plt.legend(['l2-reg FFNN', 'l2-reg CNN'], loc='upper right')
        elif (plot_index == 4):
            plt.xlabel('Validation Data')
            plt.ylabel('Classification_Error_Rate')
            plt.legend(['l2-reg FFNN', 'l2-reg CNN'], loc='upper right')
        elif (plot_index == 5):
            plt.xlabel('Batches')
            plt.ylabel('Cross_Entropy_Loss')
            plt.legend(['Dropout FFNN', 'Dropout CNN'], loc='upper right')
        elif (plot_index == 6):
            plt.xlabel('Validation Data')
            plt.ylabel('Classification_Error_Rate')
            plt.legend(['Dropout FFNN', 'Dropout CNN'], loc='upper right')
        else:
            print("Something went wrong with plot_index")

    # Clear the figure and axes after show
    plt.show()
    plt.clf()
    plt.cla()


def main():
    '''
    The main function
    '''
    
    # Get MNIST dataset from tensorflow datasets
    mnist = tf.keras.datasets.mnist

    # Data extraction
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Normalize the image dataset
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 1) Model without regularization from A03Q3
    print("A03Q3 Modeling without regularization...")
    base_model = Neural_Network(x_train, y_train, x_test, y_test)
    base_model.build_FFNN()
    base_model.compile()
    base_model.fit()
    base_model.evaluate()

    # 2) Model with l2 regularization from A03Q4
    print("A03Q4 Modeling with l2 regularization...")
    l2_model = Neural_Network(x_train, y_train, x_test, y_test)
    l2_model.build_l2norm_FFNN()
    l2_model.compile()
    l2_model.fit()
    l2_model.evaluate()

    # 3) Model with dropout regularization from A03Q4
    print("A03Q4 Modeling with dropout regularization...")
    dropout_model = Neural_Network(x_train, y_train, x_test, y_test)
    dropout_model.build_dropout_FFNN()
    dropout_model.compile()
    dropout_model.fit()
    dropout_model.evaluate()

    # 4) CNN Model without regularization
    print("Modeling with Convolutional layer without regularization...")
    cnn_model = Neural_Network(x_train.reshape(-1, 28, 28, 1), y_train, x_test.reshape(-1, 28, 28, 1), y_test)
    cnn_model.build_CNN()
    cnn_model.compile()
    cnn_model.fit()
    cnn_model.evaluate()

    # 5) CNN Model with l2 regularization
    print("Modeling with Convolutional layer and l2 regularization...")
    l2_cnn_model = Neural_Network(x_train.reshape(-1, 28, 28, 1), y_train, x_test.reshape(-1, 28, 28, 1), y_test)
    l2_cnn_model.build_l2norm_CNN()
    l2_cnn_model.compile()
    l2_cnn_model.fit()
    l2_cnn_model.evaluate()

    # 6) CNN Model with dropout regularization
    print("Modeling with Convolutional layer and dropout regularization...")
    dropout_cnn_model = Neural_Network(x_train.reshape(-1, 28, 28, 1), y_train, x_test.reshape(-1, 28, 28, 1), y_test)
    dropout_cnn_model.build_dropout_CNN()
    dropout_cnn_model.compile()
    dropout_cnn_model.fit()
    dropout_cnn_model.evaluate()

    # Collect data to plot
    plot_dict = {
        "BatchEndLoss on base_FFNN vs base_CNN": [
            np.array(base_model.batch_end_loss), 
            np.array(cnn_model.batch_end_loss)
        ],
        "ClassError on base_FFNN vs base_CNN": [
            1- np.array(base_model.HistoryObj.history['val_accuracy']), 
            1- np.array(cnn_model.HistoryObj.history['val_accuracy'])
        ],
        "BatchEndLoss on l2_FFNN vs l2_CNN": [
            np.array(l2_model.batch_end_loss), 
            np.array(l2_cnn_model.batch_end_loss)
        ],
        "ClassError on l2_FFNN vs l2_CNN": [
            1- np.array(l2_model.HistoryObj.history['val_accuracy']), 
            1- np.array(l2_cnn_model.HistoryObj.history['val_accuracy'])
        ],
        "BatchEndLoss on dropout_FFNN vs dropout_CNN": [
            np.array(dropout_model.batch_end_loss), 
            np.array(dropout_cnn_model.batch_end_loss)
        ],
        "ClassError on dropout_FFNN vs dropout_CNN": [
            1- np.array(dropout_model.HistoryObj.history['val_accuracy']), 
            1- np.array(dropout_cnn_model.HistoryObj.history['val_accuracy'])
        ]
    }

    # # Plot the diagram
    plot_diagram(plot_dict, nrows=3, ncols=2)


if __name__ == "__main__":
    # Supress the tensorflow warning messages
    # Adjust the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:
    #     0 = all messages are logged(default behavior)
    #     1 = INFO messages are not printed
    #     2 = INFO and WARNING messages are not printed
    #     3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Run the main function
    main()
