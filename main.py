import pandas as pd
import numpy as np
from numpy import argmax
from Preprocessing import Preprocessing
from Models import Models
import os
import time
import matplotlib.pyplot as plt
from keras import callbacks
from keras import optimizers
from keras.models import Model
from keras.models import load_model
from Distance import Distance
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import backend as K
np.random.seed(12)
from tensorflow import set_random_seed
import tensorflow as tf

set_random_seed(12)
PYTHONHASHSEED=0
tf.random.set_random_seed(12)
import sys

def getXY(train, test):
    clssList = train.columns.values
    print(clssList)
    target = [i for i in clssList if i.startswith('classification')]
    print(target)
    train_Y=train[target]
    # print(train_Y.head)
    test_Y=test[target]
    # remove label from dataset
    train_X = train.drop(target, axis=1)
    train_X=train_X.values
    #print(train_X.columns.values)
    test_X = test.drop(target, axis=1)
    test_X=test_X.values

    return train_X, train_Y, test_X, test_Y

def printPlotLoss(history, d):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("plotLoss" + str(d) + ".png")
    plt.close()
    # plt.show()


def printPlotAccuracy(history, d):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("plotAccuracy" + str(d) + ".png")
    plt.close()
    # plt.show()


def main():
    PREPROCESSING =0
    LOAD_AUTOENCODER=1
    CLUSTER=1
    LOAD_MODEL=0
    VALIDATION_SPLIT=.1
    prp=Preprocessing()
    NUM_CLASSES = 2
    img_rows, img_cols = 3, 10
    model=Models(NUM_CLASSES, img_rows, img_cols)
    distCls=Distance()
    path = "dataset"
    pathModels="models"
    pathResults='results'
    #CLUSTERS=50

    CLUSTERS = sys.argv[1]
   # pathOutput = sys.argv[2]




    col_names = np.array(["duration", "protocol_type", "service", "flag", "src_bytes",
                          "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                          "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                          "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                          "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                          "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                          "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                          "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                          "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                          "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "classification."])
    if (PREPROCESSING == 1):
        tic = time.time()  # recupera il tempo corrente in secondi
        train_df = pd.read_csv(os.path.join(path, "KDDTrain+aggregate.csv"), delimiter=",")
        test_df = pd.read_csv(os.path.join(path,"KDDTest+aggregate.csv"), delimiter=",")

        nominal_inx = [1, 2, 3]
        binary_inx = [6, 11, 13, 14, 20, 21]
        numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))

        nominal_cols = col_names[nominal_inx].tolist()
        # binary_cols = col_names[binary_inx].tolist()
        numeric_cols = col_names[numeric_inx].tolist()

        train_df, test_df = prp.ohe(train_df, test_df, nominal_cols)
        train_df, test_df = prp.standardScale(train_df, test_df, numeric_cols)
        # copyOfTrain = minMaxScale(copyOfTrain, numeric_cols)
        train_df["classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0), inplace=True)
        test_df["classification."].replace(to_replace=dict(normal=1, R2L=0, Dos=0, Probe=0, U2R=0), inplace=True)

        train_df.to_csv(os.path.join(path, 'DatasetStand','Train_standard.csv'), index=False)
        # copyOfTrain = minMaxScale(copyOfTrain, numeric_cols)

        test_df.to_csv(os.path.join(path, 'DatasetStand','Test_standard.csv'), index=False)
        toc=time.time()
        timePreprocessing= toc-tic
    else:
        train_df=pd.read_csv(os.path.join(path, 'DatasetStand', 'Train_standard.csv'))
        test_df= pd.read_csv(os.path.join(path, 'DatasetStand', 'Test_standard.csv'))

    print("Train shape: ", train_df.shape)
    print("Test shape: ", test_df.shape)
    
    train_X, train_Y, test_X, test_Y = getXY(train_df, test_df)


    if (LOAD_AUTOENCODER==0):
        tic=time.time()
        callbacks_list = [
            # callbacks.ModelCheckpoint(
            #   filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            #  monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=4, restore_best_weights=True),
            # reduce_lr
        ]

        print('Model with autoencoder')
        # parametri per autoencoder
        p = {
            'first_layer': 80,
            'second_layer': 30,
            'third_layer': 10,
            'batch_size': 64,
            'epochs': 150,
            'optimizer': optimizers.Adam,
            'kernel_initializer': 'glorot_uniform',
            'losses': 'mse',
            'first_activation': 'tanh',
            'second_activation': 'tanh',
            'third_activation': 'sigmoid'}

        autoencoder = model.autoencoder(train_X, p)
        autoencoder.summary()

        # estract encoder layers from autoEncoder
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder3').output)
        encoder.summary()



        history = autoencoder.fit(train_X, train_X,
                                  validation_split=VALIDATION_SPLIT,
                                  batch_size=p['batch_size'],
                                  epochs=p['epochs'], shuffle=False,
                                  callbacks=callbacks_list,
                                  verbose=1)


        toc=time.time()
        timeAutoencoder=toc-tic
        printPlotAccuracy(history, 'autoencoder')
        printPlotLoss(history, 'autoencoder')
        modelA='autoencoder.h5'
        autoencoder.save(os.path.join(pathModels, modelA))
    else:
        print("Load autoencoder from disk")
        modelA = 'autoencoder.h5'
        autoencoder = load_model(os.path.join(pathModels, modelA))
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder3').output)
        encoder.summary()
        # plot_model(autoencoder, to_file='autoencoder.png')

    encoded_train = pd.DataFrame(encoder.predict(train_X))
    encoded_train = encoded_train.add_prefix('feature_')
    encoded_train["classification"] = train_Y
    print(encoded_train.shape)
    print(encoded_train.head())
    encoded_train.to_csv(os.path.join(path,'train_encoded.csv'), index=False)
    encoded_test = pd.DataFrame(encoder.predict(test_X))
    encoded_test = encoded_test.add_prefix('feature_')
    encoded_test["classification"] = test_Y
    encoded_test.to_csv(os.path.join(path,'test_encoded.csv'), index=False)


    if (CLUSTER==1):
        tic=time.time()
        test = pd.read_csv(os.path.join(path,'test_encoded.csv'), delimiter=",")
        train = pd.read_csv(os.path.join(path,'train_encoded.csv'), delimiter=",")

        trainNormal = train[train['classification'] == 1]
        trainAttack = train[train['classification'] == 0]

        testNormal = test[test['classification'] == 1]
        testAttack = test[test['classification'] == 0]

        trainY = train['classification']
        trainY = trainY.sort_values(ascending=False)
        trainY = pd.DataFrame(trainY)
        TrainY = trainY.values

        testY = test['classification']
        testY = testY.sort_values(ascending=False)
        testY = pd.DataFrame(testY)
        testY = testY.values
        '''
        distTrain = findNearest(train, trainNormal, trainAttack)
        distTrain=np.array(distTrain)
        '''
        trainX, trainY,  testX, testY= getXY(train, test)

        print("Train shape after encode: ", trainX.shape )
        print("Test shape after encode: ", testX.shape)


        distTrain, distTest = distCls.distance(trainX, testX, trainNormal, trainAttack, int(CLUSTERS), False)
        #distTrain, distTest = distCls.distanceNOCls(trainX, testX, trainNormal, trainAttack)
        distTrain = np.array(distTrain)
        distTest = np.array(distTest)
        '''
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(toimage(distTest[i]))
        plt.savefig('imagetest.png')
        '''


        # saveNpArray(dist, Y)
        # saveNpArray(distTrain, trainY, "Train")
        print('Saving dataset')
        distCls.saveNpArray(distTrain, trainY, "C"+str(CLUSTERS)+"Train")
        distCls.saveNpArray(distTest, testY, "C"+str(CLUSTERS)+"Test")


        toc = time.time()
        time_clustering = toc - tic

    x_train = np.load(os.path.join(path, "C"+str(CLUSTERS)+"TrainX.npy"))
    y_train = np.load(os.path.join(path,"C"+str(CLUSTERS)+ "TrainY.npy"))
    x_test = np.load(os.path.join(path, "C"+str(CLUSTERS)+"TestX.npy"))
    y_test = np.load(os.path.join(path, "C"+str(CLUSTERS)+"TestY.npy"))
    batch_size = 64
    num_classes = 2
    epochs = 150
    img_rows, img_cols = 3, 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test1 = np_utils.to_categorical(y_test, num_classes)
    print("train y after: ", y_train.shape)
    print("test x:", x_test.shape)
    print("test y after: ", y_test1.shape)

    if(LOAD_MODEL==0):
        tic=time.time()




        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True),
        ]



        # model = load_model('ModelCNN\modelNoPoll.h5')
        # model = load_model('dataset\ModelCNN\modelPolling.h5')
        model = model.baselineCNN(x_train)
        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_test, y_test1),
                  callbacks=callbacks_list
                  )
        modelName = str(CLUSTERS) + 'cnn.h5'
        model.save(os.path.join(pathModels, modelName))
        toc = time.time()
        time_classifier = toc - tic
    else:
        modelName = str(CLUSTERS) + 'cnn.h5'
        model=load_model(os.path.join(pathModels,modelName))
        model.summary()

        # score = model.evaluate(x_test, y_test, verbose=0)
    print("train y after: ", y_train.shape)
    print("test x:", x_test.shape)
    print("test y after: ", y_test1.shape)

    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    print(y_pred.shape)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred, normalize=True)
    print(cm)
    print(acc)

    fileOutput=str(CLUSTERS)+'result.txt'

    file = open(os.path.join(pathResults, fileOutput), 'w')
    file.write('Confusion matrix for: %s clusters \n' %CLUSTERS)
    file.write(str(cm))
    file.write('\n')
    file.write(str(acc))
    file.write('\n')







    if (PREPROCESSING == 1):
        print("Time for preprocessing 1 %s " % timePreprocessing)
        file.write('Time for preprocessing 1 %s  \n' % timePreprocessing)

    if (LOAD_AUTOENCODER == 0):
        print("Time for train autoencoder 1 %s " % timeAutoencoder)
        file.write("Time for train autoencoder 1 %s \n " % timeAutoencoder)

    if (CLUSTER == 1):
        print("Time for preprocessing clustering %s " % time_clustering)
        file.write("Time for preprocessing clustering %s \n" % time_clustering)
    if (LOAD_MODEL== 0):
        print("Time for train classifier %s " % time_classifier)
        file.write("Time for train classifier %s  \n" % time_classifier)






if __name__ == "__main__":
    main()
