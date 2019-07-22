import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import tensorflow as tf

SEED = 123                 # to be able to rerun the same NN
np.random.seed(SEED)
tf.set_random_seed(SEED)

np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

get_ipython().run_line_magic('matplotlib','inline')
import scipy.io as sio
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import f1_score

def test(filename):
    from keras.models import load_model
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from scipy.misc import imresize
    model = load_model('my_model.h5')
    model.load_weights('my_model_weights.h5')
    global classes
    classes = [0,1,2,3,4,5,6,7,8,9]
    image=filename
    img_plt=mpimg.imread(image)
    img=cv2.imread(image)
    image_size=(32,32)
    img=imresize(img,image_size)
    img = np.expand_dims(img, axis=0)
    arr=model.predict(img)
    index=np.where(arr==1.0000)[1]
    predicted=classes[index[0]]
    print('Predicted Number: %d' % predicted)
    plt.imshow(img_plt)
    plt.show()
    
    
def traintest():
    
    import urllib.request

    get_ipython().run_line_magic('mkdir', 'data')

    urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "data/train_32x32.mat")
    urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "data/test_32x32.mat")
    urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "data/extra_32x32.mat")

    train_data = sio.loadmat('data/train_32x32.mat')
    test_data = sio.loadmat('data/test_32x32.mat')
    extra_data = sio.loadmat('data/extra_32x32.mat')


    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    X_extra, y_extra = extra_data['X'], extra_data['y']

    global classes
    classes = [0,1,2,3,4,5,6,7,8,9]
    nb_classes = 10
    
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    y_extra[y_extra == 10] = 0

    #print(X_train.shape, X_test.shape, X_extra.shape)
    
    X_train = np.transpose(X_train,(3,0,1,2))
    X_test = np.transpose(X_test,(3,0,1,2))
    X_extra = np.transpose(X_extra,(3,0,1,2))

    X_train = np.concatenate([X_train, X_extra])
    y_train = np.concatenate([y_train, y_extra])

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #y_train[:4]
    
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(len(y_train[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',      
              metrics=['accuracy'])
    
    #model.summary()
    model_history = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split = 0.1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    
    
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'# returns a compiled model
    model.save_weights('my_model_weights.h5')
    
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred,axis=1)
    type(y_test)
    a=test_data['y']
    return f1_score(a, y_pred_bool,average='micro')
