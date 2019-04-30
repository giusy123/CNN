from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras import optimizers
from keras.models import Model
from keras import regularizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import LeakyReLU

np.random.seed(12)
from tensorflow import set_random_seed

set_random_seed(12)


class Models():
    def __init__(self, n_classes, img_rows, img_cols):
        self._nClass = n_classes
        self._img_rows=img_rows
        self._img_cols=img_cols


    def baselineCNN(self, train):

        input_shape = (self._img_rows, self._img_cols, 1)
        '''
        inputs = Input(shape=(input_shape))
        y = Conv2D(filters=32,
                   kernel_size=(2),
                   activation='relu')(inputs)
        y=Dropout(0.3)(y)
        y = Conv2D(filters=16,
                   kernel_size=(2,4),
                   activation='relu')(y)
        y = Dropout(0.3)(y)
        y=Flatten()(y)
        outputs=Dense(self._nClass,activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)

        #Creazione primo modello senza pooling
        '''
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(2),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Conv2D(16, kernel_size=(2, 4), activation='relu')),
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(self._nClass, activation='softmax'))

        model.summary()
        return model

    # creazione secondo modello con pooling
    def CNNPooling(self):
        input_shape = (self._img_rows, self._img_cols, 1)
        model1 = Sequential()

        model1.add(Conv2D(32, kernel_size=(2),
                         activation='relu',
                         input_shape=input_shape))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Dropout(0.25))
        model1.add(Conv2D(64, kernel_size=(1, 3), activation='relu')),
        model1.add(Flatten())
        model1.add(Dropout(0.5))
        model1.add(Dense(20, activation='relu'))
        model1.add(Dense(self._nClass, activation='softmax'))

        model1.save('modelPooling.h5')
        model1.summary()
        return model1

    def secondModelCNN(self, x_train):
        kernel_size = (1, 2)
        filters = 64
        dropout = 0.3
        num_labels = 2
        image_size = x_train.shape[1]
        input_shape = (3, 10, 1)

        inputs = Input(shape=(input_shape))
        y = Conv2D(filters=8,
                   kernel_size=(2, 2),
                   activation='relu')(inputs)
        # y = MaxPooling2D()(y)
        y = Dropout(0.5)(y)
        y = Conv2D(filters=8,
                   kernel_size=(2,3),
                   activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Conv2D(filters=8,
                   kernel_size=(1,4),
                   activation='relu')(y)
        y = Dropout(0.5)(y)
        #  y = MaxPooling2D()(y)
        # image to vector before connecting to dense layer
        y = BatchNormalization()(y)
        y = Flatten()(y)

        # dropout regularization
        y = Dropout(dropout)(y)
        outputs = Dense(num_labels, activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        # network model in text
        model.summary()
        return model

    def autoencoder(self, x_train, params):
        # A deep autoecndoer model
        n_col = x_train.shape[1]
        input = Input(shape=(n_col,))
        print(input)
        # encoder_layer
        # Dropoout?
        #  input1 = Dropout(.2)(input)
        encoded = Dense(params['first_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],
                        name='encoder1')(input)
        encoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],

                        name='encoder2')(encoded)
        encoded = Dense(params['third_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],

                        name='encoder3')(encoded)

        # l1 = BatchNormalization()(encoded)
        # encoded = Dropout(.5)(encoded)
        decoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'], name='decoder1')(encoded)
        decoded = Dense(params['first_layer'], activation=params['second_activation'],
                        kernel_initializer=params['kernel_initializer'], name='decoder2')(decoded)
        decoded = Dense(n_col, activation=params['third_activation'],
                        kernel_initializer=params['kernel_initializer'], name='decoder')(decoded)
        # serve per L2 normalization?
        # encoded1_bn = BatchNormalization()(encoded)

        autoencoder = Model(input=input, output=decoded)

        learning_rate = 0.001
        decay = learning_rate / params['epochs']
        autoencoder.compile(loss=params['losses'],
                            optimizer=params['optimizer']()
                            # (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, amsgrad=False)#
                            , metrics=['accuracy'])

        return autoencoder