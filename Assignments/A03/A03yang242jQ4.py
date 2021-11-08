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


def plot_CEL_on_Batch(l2_CEL, dropout_CEL, se_CEL, CEL, nrows, ncols, index):
    plt.subplot(nrows, ncols, index)
    plt.plot(l2_CEL)
    plt.plot(dropout_CEL)
    plt.plot(se_CEL)
    plt.plot(CEL)
    plt.title('Cross Entropy Loss on the Batches')
    plt.ylabel('Cross_Entropy_Loss')
    plt.xlabel('Batches')
    plt.legend(['l2-regularization', 'Dropout',
               'Early stopping', 'Q3 No regularization'], loc='upper right')


def plot_ClassError_on_Valid(l2_CLE, dropout_CLE, se_CLE, CLE, nrows, ncols, index):
    plt.subplot(nrows, ncols, index)
    plt.plot(l2_CLE)
    plt.plot(dropout_CLE)
    plt.plot(se_CLE)
    plt.plot(CLE)
    plt.title('Classification Error Rate on Validation data')
    plt.ylabel('Classification_Error_Rate')
    plt.xlabel('Validation Data')
    plt.legend(['l2-regularization', 'Dropout',
               'Early stopping', 'Q3 No regularization'], loc='upper right')


def build_l2_model():
    # Define the model structure,
    # Input layer:          Flatten the normalized inout image
    # First hidden layer:   Dimension of 500, applied with relu activation function
    #                       applied with l2 regularization
    # Second hidden layer:  Dimension of 500, applied with relu activation function,
    #                       applied with l2 regularization
    # Output layer:         10 predict labels, applied with softmax activation function,
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(500, activation='relu',
                              kernel_regularizer='l2',
                              bias_regularizer='l2'),
        tf.keras.layers.Dense(500, activation='relu',
                              kernel_regularizer='l2',
                              bias_regularizer='l2'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with Stocastic Gradient Descent optimizer,
    #                        Sparse Categorical Cross Entropy Loss,
    #                        output the accuracy of prediction
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_dropout_model():
    # Define the model structure,
    # Input layer:          Flatten the normalized inout image
    # Dropout layer:        Randomly sets input units to 0 with a frequency of rate 0.2
    # First hidden layer:   Dimension of 500, applied with relu activation function
    # Dropout layer:        Randomly sets input units to 0 with a frequency of rate 0.2
    # Second hidden layer:  Dimension of 500, applied with relu activation function
    # Dropout layer:        Randomly sets input units to 0 with a frequency of rate 0.2
    # Output layer:         10 predict labels, applied with softmax activation function
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with Stocastic Gradient Descent optimizer,
    #                        Sparse Categorical Cross Entropy Loss,
    #                        output the accuracy of prediction
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_model():
    # Define the model structure,
    # Input layer:          Flatten the normalized inout image
    # First hidden layer:   Dimension of 500, applied with relu activation function
    # Second hidden layer:  Dimension of 500, applied with relu activation function
    # Output layer:         10 predict labels, applied with softmax activation function
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


def fit_eva_model(model, training_x, training_y, testing_x, testing_y):
    batch_end_loss = list()

    class MyCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            batch_end_loss.append(logs['loss'])

    HistoryObj = model.fit(
        training_x,
        training_y,
        batch_size=128,
        epochs=250,
        validation_split=0.1,
        callbacks=[MyCallback()]
    )

    print("Model Evaluating:")
    model.evaluate(testing_x,  testing_y, verbose=2)
    model.summary()

    return batch_end_loss, HistoryObj


def fit_eva_earlystopping_model(model, training_x, training_y, testing_x, testing_y):
    batch_end_loss = list()

    class MyCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            batch_end_loss.append(logs['loss'])

    EarlyStopping_Callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=2, patience=3)

    HistoryObj = model.fit(
        training_x,
        training_y,
        batch_size=128,
        epochs=250,
        validation_split=0.1,
        callbacks=[
            MyCallback(),
            EarlyStopping_Callback
        ]
    )

    print("Early Stopping Model Evaluating:")
    model.evaluate(testing_x,  testing_y, verbose=2)
    model.summary()

    return batch_end_loss, HistoryObj


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

    # Model with l2 regularization
    print("Modeling with l2 regularization...")
    l2_model = build_l2_model()
    l2_batch_end_loss, l2_HistoryObj = fit_eva_model(
        l2_model, x_train, y_train, x_test, y_test)

    # Model with dropout regularization
    print("Modeling with dropout regularization...")
    dropout_model = build_dropout_model()
    dropout_batch_end_loss, dropout_HistoryObj = fit_eva_model(
        dropout_model, x_train, y_train, x_test, y_test)

    # Model with early stopping regularization
    print("Modeling with early stopping regularization...")
    es_model = build_model()
    es_batch_end_loss, es_HistoryObj = fit_eva_earlystopping_model(
        es_model, x_train, y_train, x_test, y_test)

    # Model without regularization from Q3
    print("Q3 Modeling without regularization...")
    model = build_model()
    batch_end_loss, HistoryObj = fit_eva_model(
        model, x_train, y_train, x_test, y_test)

    # Define plot figure size, 2_inch_width by 1_inch_height
    plt.figure(figsize=(10, 5))

    # Plot the cross entropy loss on the batches
    l2_CEL_on_Batch = l2_batch_end_loss
    dropout_CEL_on_Batch = dropout_batch_end_loss
    es_CEL_on_Batch = es_batch_end_loss
    CEL_on_Batch = batch_end_loss
    plot_CEL_on_Batch(l2_CEL_on_Batch, dropout_CEL_on_Batch,
                      es_CEL_on_Batch, CEL_on_Batch, 1, 2, 1)
    # Plot the classification error on the validation data, with l2-reg
    l2_CLE_on_val = [1-x for x in l2_HistoryObj.history['val_accuracy']]
    dropout_CLE_on_val = [1-x for x in dropout_HistoryObj.history['val_accuracy']]
    es_CLE_on_val = [1-x for x in es_HistoryObj.history['val_accuracy']]
    CLE_on_val = [1-x for x in HistoryObj.history['val_accuracy']]
    plot_ClassError_on_Valid(
        l2_CLE_on_val, dropout_CLE_on_val, es_CLE_on_val, CLE_on_val, 1, 2, 2)

    # Clear the figure and axes after show
    plt.show()
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
