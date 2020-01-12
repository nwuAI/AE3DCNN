import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import dot, concatenate, Lambda, RepeatVector, Add, Multiply
import numpy
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import keras.optimizers as optimizers
from tt import focal_loss

K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 160, 160, 18
videoopath = '../../datasets/SMIC_TIM10/'
flowpath = '../../datasets/SMIC_DEEPFLOW10/'
tot_mat = numpy.zeros((3, 3))
for sub in range(1, 21):
    print(".starting subject" + str(sub))
    subname = 's' + str(sub) + '_'
    training_list = []
    testing_list = []
    training_labels = []
    testing_labels = []
    negnum, posnum, surnum = 0, 0, 0
    emotionlisting = os.listdir(videoopath)
    for emotiondata in emotionlisting:  # neg,pos,sur
        negativepath = videoopath + emotiondata + "/"
        negflowepath = flowpath + emotiondata + "/"
        directorylisting = os.listdir(negativepath)
        directoryflowlisting = os.listdir(negflowepath)
        i = 0
        for video in directorylisting:
            videoflowpath = negflowepath + directoryflowlisting[i]
            i = i + 1
            videopath = negativepath + video
            frames = []
            framelisting = os.listdir(videopath)
            frameflowlisting = os.listdir(videoflowpath)
            framerange = [x for x in range(10)]
            frameflowrange = [x for x in range(8)]
            for frame in framerange:
                imagepath = videopath + "/" + framelisting[frame]
                image = cv2.imread(imagepath)
                # imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
            for frame in frameflowrange:
                imagepath = videoflowpath + "/" + frameflowlisting[frame]
                image = cv2.imread(imagepath)
                # imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
            frames = numpy.asarray(frames)
            videoarray = frames  # numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            if subname in video:
                testing_list.append(videoarray)
                if 'Negative' in emotiondata:
                    testing_labels.append(0)
                elif 'Positive' in emotiondata:
                    testing_labels.append(1)
                else:
                    testing_labels.append(2)
            else:
                training_list.append(videoarray)
                if 'Negative' in emotiondata:
                    training_labels.append(0)
                    negnum += 1
                elif 'Positive' in emotiondata:
                    training_labels.append(1)
                    posnum += 1
                else:
                    training_labels.append(2)
                    surnum += 1

    # -------------------------------------------------------------------------------------------
    training_list = numpy.asarray(training_list)
    trainingsamples = len(training_list)
    testing_list = numpy.asarray(testing_list)
    testingsamples = len(testing_list)
    if testingsamples == 0:
        continue
    testing_labels = numpy.asarray(testing_labels)
    training_labels = numpy.asarray(training_labels)
    training_labels = np_utils.to_categorical(training_labels, 3)
    testing_labels = np_utils.to_categorical(testing_labels, 3)

    training_set = numpy.zeros((trainingsamples, 1, image_depth, image_rows, image_columns))
    testing_set = numpy.zeros((testingsamples, 1, image_depth, image_rows, image_columns))
    for h in range(trainingsamples):
        training_set[h][0][:][:][:] = training_list[h, :, :, :]
    for h in range(testingsamples):
        testing_set[h][0][:][:][:] = testing_list[h, :, :, :]
    training_set = training_set.astype('float32')
    training_set -= numpy.mean(training_set)
    training_set /= numpy.max(training_set)
    testing_set = testing_set.astype('float32')
    testing_set -= numpy.mean(testing_set)
    testing_set /= numpy.max(testing_set)
    # training_set = training_set / 255
    # testing_set = testing_set / 255
    inputs = Input(shape=(1, image_depth, image_rows, image_columns))
    videorow = Lambda(lambda x: x[:, :, 0:10, :, :])(inputs)  # 64,64,10
    flowrow = Lambda(lambda x: x[:, :, 10:18, :, :])(inputs)  # 64,64,9
    # conv-pooling

    # 3d
    flowconv1 = Convolution3D(16, (1, 3, 3), activation='relu')(flowrow)  # 16,8,158,158
    flowmax1 = MaxPooling3D(pool_size=(1, 3, 3))(flowconv1)  # 16,8,52,52
    flowconv2 = Convolution3D(16, (1, 3, 3), activation='relu')(flowmax1)  # 16,8,50,50
    flowmax2 = MaxPooling3D(pool_size=(1, 3, 3))(flowconv2)  # 16,8,16,16
    flowmax2 = Dropout(0.5)(flowmax2)

    videoconv1 = Convolution3D(16, (1, 3, 3), activation='relu')(videorow)
    videomax1 = MaxPooling3D(pool_size=(1, 3, 3))(videoconv1)
    videoconv2 = Convolution3D(16, (1, 3, 3), activation='relu')(videomax1)
    videomax2 = MaxPooling3D(pool_size=(1, 3, 3))(videoconv2)
    videomax2 = Dropout(0.5)(videomax2)

    flow_data = Reshape(target_shape=(16, 8, 256))(flowmax2)
    flow_data = Dense(25)(flow_data)  # (1,8,25)
    flow_data = Reshape(target_shape=(16, 200))(flow_data)
    flow_att_1 = Activation('tanh')(flow_data)
    flow_att_1 = Dense(2)(flow_att_1)  # (16,2)
    flow_att_1 = Permute((2, 1), input_shape=(16, 2))(flow_att_1)  # 2,16
    flow_att_1 = Activation('softmax')(flow_att_1)
    flow_new = dot([flow_att_1, flow_data], axes=[-1, 1])  # 2,200

    video_data = Reshape(target_shape=(16, 10, 256))(videomax2)
    video_data = Dense(25)(video_data)  # (1,10,25)
    video_data = Reshape(target_shape=(16, 250))(video_data)
    video_att_1 = Activation('tanh')(video_data)
    video_att_1 = Dense(2)(video_att_1)  # (16,2)
    video_att_1 = Permute((2, 1), input_shape=(16, 2))(video_att_1)
    video_att_1 = Activation('softmax')(video_att_1)
    video_new = dot([video_att_1, video_data], axes=[-1, 1])  # 2,250

    # SOFTMAX
    concat1 = concatenate([flow_new, video_new], axis=-1)
    flatten1 = Flatten()(concat1)  # 900
    drop1 = Dropout(0.5)(flatten1)
    dense1 = Dense(32, kernel_initializer='normal', activation='relu')(flatten1)  # (3)
    drop2 = Dropout(0.5)(dense1)
    dense2 = Dense(3, kernel_initializer='normal', activation='relu')(drop2)  # (3)
    out = Activation('softmax')(dense2)
    model = Model(inputs=[inputs], outputs=out)
    classes_num = [negnum, posnum, surnum]
    sgd = SGD(lr=0.005, decay=0.0005, momentum=0.9)
    # sgd = optimizers.SGD(lr=0.0009, decay=0.00009, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss=[focal_loss(classes_num)], optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    filepath = "weights_microexpstcnn/SMIC_LOSOsub" + str(sub) + "_best3d.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2, mode='min', restore_best_weights=True)
    # EarlyStopping(monitor='val_loss', patience=20 verbose=2)
    callbacks_list = [checkpoint, early_stopping]
    # numpy.save("numpy_training_datasets/SMIC_LOSOsub" + str(sub) + "_val_images.npy", training_set)
    # numpy.save("numpy_training_datasets/SMIC_LOSOsub" + str(sub) + "val_labels.npy", training_labels)
    # numpy.save("numpy_validation_dataset/SMIC_LOSOsub" + str(sub) + "_val_images.npy", testing_set)
    # numpy.save("numpy_validation_dataset/SMIC_LOSOsub" + str(sub) + "val_labels.npy", testing_labels)
    hist = model.fit(training_set, training_labels, validation_data=(testing_set, testing_labels),
                     callbacks=callbacks_list, batch_size=8, nb_epoch=100, shuffle=True)
    # predict
    # print("predict subject" + str(sub))

    model = load_model(filepath)
    predict = model.predict(testing_set, batch_size=16)
    # print(predict)
    print("cm subject" + str(sub))
    testing_labels = numpy.argmax(testing_labels, axis=1)
    predict = numpy.argmax(predict, axis=1)
    ct = confusion_matrix(testing_labels, predict)
    print(ct)
    # check the order of the CT
    order = numpy.unique(numpy.concatenate((predict, testing_labels)))
    # create an array to hold the CT for each CV
    mat = numpy.zeros((3, 3))

    # put the order accordingly, in order to form the overall ConfusionMat
    for m in range(len(order)):
        for n in range(len(order)):
            mat[int(order[m]), int(order[n])] = ct[m, n]

    tot_mat = mat + tot_mat
    print(tot_mat)
    if sub == 20:
        # microAcc = numpy.trace(tot_mat) / numpy.sum(tot_mat)
        [f1, precision, recall] = fpr(tot_mat, 3)
        war = weighted_average_recall(tot_mat, 3, 164)
        uar = unweighted_average_recall(tot_mat, 3)
        print("f1:" + str(f1))
        print("war: " + str(war))
        print("uar: " + str(uar))
        print("cm:")
        print(tot_mat)



