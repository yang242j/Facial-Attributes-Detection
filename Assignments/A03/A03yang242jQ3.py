# =============================================================================
# #
# #  Created on Thu. Oct. 21st 2021
# #
# #  ENSE 885AU: A03-Q3
# #
# #  @author: Jingkang Yang <yang242j@uregina.ca>
# #
# #  Execution Command: python A03yang242jQ3.py
# #
# #  Question Declaration:
# #
# #     Build a three layer feedforward network
# #     using Tensorflow on MNIST dataset
# #     1) The hidden layer h1 and h2 have dimension 500
# #     2) Train the network for 250 epochs
# #     3) Test the classification error
# #     4) No regularization
# #     5) Plot the cross entropy loss on the batches
# #     6) Plot the classification error on the validation data
# #
# =============================================================================
# IMPORTS
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_CEL_on_Batch(CEL_on_Batch, plot_index):
    plt.subplot(1, 2, plot_index)
    plt.plot(CEL_on_Batch)
    plt.title('Cross Entropy Loss on the Batches')
    plt.ylabel('Cross_Entropy_Loss')
    plt.xlabel('Batches')


def plot_ClassError_on_Valid(CLE, plot_index):
    plt.subplot(1, 2, plot_index)
    plt.plot(CLE)
    plt.title('Classification Error Rate on Validation data')
    plt.ylabel('Classification_Error_Rate')
    plt.xlabel('Validation Data')
    # plt.legend(['train', 'validation'], loc='upper left')


def build_model():
    # Define the model structure,
    # Input layer:          Flatten the normalized inout image
    # First hidden layer:   Dimension of 500, applied with relu activation function
    # Second hidden layer:  Dimension of 500, applied with relu activation function
    # Output layer:         10 predict labels, applied with softmax activation function,
    #                       converted to probabilities of 10 lables
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with Stocastic Gradient Descent optimizer,
    #                        Sparse Categorical Cross Entropy Loss,
    #                        output the accuracy of prediction
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    return model


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

    # Build the model
    model = build_model()

    # Save the loss after each mini-batch, instead of saving the loss after each epochs
    batch_end_loss = list()
    class SaveBatchLoss(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            batch_end_loss.append(logs['loss'])

    # Train or fit the model with the training images and labels,
    # using mini-batch of size 128 image per batch,
    # train the model for 250 epochs,
    # split 10% of the training sets as the validation sets
    HistoryObj = model.fit(
        x_train,
        y_train, 
        batch_size=128,
        epochs=250,
        validation_split=0.1,
        callbacks=[SaveBatchLoss()])

    print("Evaluating:")
    model.evaluate(x_test,  y_test, verbose=1)
    model.summary()

    # Define plot figure size, 2_inch_width by 1_inch_height
    plt.figure(figsize=(10, 5))
    # Plot the cross entropy loss on the batches
    CEL_on_Batch = batch_end_loss
    plot_CEL_on_Batch(CEL_on_Batch, 1)
    # Plot the classification error on the validation data
    CLE_on_val = [1-x for x in HistoryObj.history['val_accuracy']]
    plot_ClassError_on_Valid(CLE_on_val, 2)
    plt.show()
    # Clear the figure and axes
    plt.clf()
    plt.cla()


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
