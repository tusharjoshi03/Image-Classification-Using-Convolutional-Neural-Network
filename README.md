# Street View House Number Image Classification Using CNN

This project aims to classify images from Google's SVHN dataset into ten different classes of numbers (0-9) in order to identify house numbers from the given input images.

The algorithm used for this classification is Convolutional Neural Network algorithm, which is implemented using Python Machine Learning library and TensorFlow framework.

The code is divided into two files, harness.ipynb and svhn.py. The file harness.ipynb contains main function, which invokes functions to train the model from scratch and then use the trained model to identify house number from a given input image. The file svhn.py is a python script, which has two functions, traintest() and test().

A model trained using traintest() function is saved as my_model.h5 and the final weights of the neural network layers are stored in my_model_weights.h5 file. These files are used to restore the model state when new unseen images are given as input to the model for classification.

The details of CNN architecture including convolutional layers, pooling layer and fully connected layer are given in the project report file. This model achieved an F1 score of 0.92 (92%) when evaluated using test dataset.
