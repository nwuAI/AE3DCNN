import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import dot, concatenate, Lambda, RepeatVector
import numpy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras import optimizers

K.set_image_dim_ordering('th')
import os
from keras.backend.tensorflow_backend import set_session

image_rows, image_columns, image_depth = 64, 64, 26
videoopath = '../../datasets/CASME_DETAIL_TIM10/'
flowpath = '../../datasets/CASME_DEEPFLOW10/'
tot_mat = numpy.zeros((4, 4))
for sub in range(1, 27):
    print(".starting subject" + str(sub))
    if sub < 10:
        subname = 'sub0' + str(sub) + '_'
    else:
        subname = 'sub' + str(sub) + '_'
    training_list = []
    testing_list = []
    training_labels = []
    testing_labels = []
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
            frameflowrange = [x for x in range(16)]
            for frame in framerange:
                imagepath = videopath + "/" + framelisting[frame]
                image = cv2.imread(imagepath)
                imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
            for frame in frameflowrange:
                imagepath = videoflowpath + "/" + frameflowlisting[frame]
                image = cv2.imread(imagepath)
                imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
                grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)

            if subname in video:  # disgust;0,happy:1,other:2,repression:2,sur:3
                testing_list.append(videoarray)
                if 'disgust' in emotiondata:
                    testing_labels.append(0)
                elif 'happy' in emotiondata:
                    testing_labels.append(1)
                elif 'other' in emotiondata:
                    testing_labels.append(2)
                elif 'repression' in emotiondata:
                    testing_labels.append(2)
                else:
                    testing_labels.append(3)

            else:
                training_list.append(videoarray)
                if 'disgust' in emotiondata:
                    training_labels.append(0)
                elif 'happy' in emotiondata:
                    training_labels.append(1)
                elif 'other' in emotiondata:
                    training_labels.append(2)
                elif 'repression' in emotiondata:
                    training_labels.append(2)
                else:
                    training_labels.append(3)

    # -------------------------------------------------------------------------------------------
    training_list = numpy.asarray(training_list)
    trainingsamples = len(training_list)
    testing_list = numpy.asarray(testing_list)
    testingsamples = len(testing_list)
    if testingsamples == 0:
        continue
    testing_labels = numpy.asarray(testing_labels)
    training_labels = numpy.asarray(training_labels)
    training_labels = np_utils.to_categorical(training_labels, 4)
    testing_labels = np_utils.to_categorical(testing_labels, 4)

    training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))
    testing_set = numpy.zeros((testingsamples, 1, image_rows, image_columns, image_depth))
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

    inputs = Input(shape=(1, image_rows, image_columns, image_depth))

    videorow = Lambda(lambda x: x[:, :, :, :, 0:10])(inputs)
    flowrow = Lambda(lambda x: x[:, :, :, :, 10:26])(inputs)
    flowconv1 = Convolution3D(32, (3, 3, 8), activation='relu')(flowrow)  # 62,62,9
    flowmax1 = MaxPooling3D(pool_size=(3, 3, 3))(flowconv1)  # 20,20,3
    videoconv1 = Convolution3D(32, (3, 3, 5), activation='relu')(videorow)  # 62,62,6
    videomax1 = MaxPooling3D(pool_size=(3, 3, 2))(videoconv1)  # 20,20,3
    # self-attention

    flow_data_attention = Reshape(target_shape=(32, 1200))(flowmax1)
    flow_att_1 = Activation('tanh')(flow_data_attention)
    flow_att_1 = Dense(3)(flow_att_1)
    flow_att_1 = Permute((2, 1), input_shape=(32, 3))(flow_att_1)
    flow_att_1 = Activation('softmax')(flow_att_1)
    flowdata = dot([flow_att_1, flow_data_attention], axes=[-1, 1])
    flowdata = Reshape(target_shape=(9, 400))(flowdata)

    video_data_attention = Reshape(target_shape=(32, 1200))(videomax1)
    video_att_1 = Activation('tanh')(video_data_attention)
    video_att_1 = Dense(3)(video_att_1)
    video_att_1 = Permute((2, 1), input_shape=(32, 3))(video_att_1)
    video_att_1 = Activation('softmax')(video_att_1)
    videodata = dot([video_att_1, video_data_attention], axes=[-1, 1])
    videodata = Reshape(target_shape=(9, 400))(videodata)
    # flow->video
    att_1 = concatenate([flowdata, videodata], axis=-1)
    att_1 = Reshape(target_shape=(9, 800))(att_1)
    att_1 = Activation('tanh')(att_1)
    att_1 = Dense(1)(att_1)
    att_1 = Activation('softmax')(att_1)
    att_1_pro = Flatten()(att_1)
    video_new = dot([att_1_pro, videodata], axes=(1, 1))
    # video->flow
    video_new_repeat = RepeatVector(9)(video_new)
    att_2 = concatenate([video_new_repeat, flowdata], axis=-1)
    att_2 = Activation('tanh')(att_2)
    att_2 = Dense(1)(att_2)
    att_2 = Activation('softmax')(att_2)
    att_2_pro = Flatten()(att_2)
    flow_new = dot([att_2_pro, flowdata], axes=(1, 1))

    video_flow_new = concatenate([video_new, flow_new], axis=-1)

    drop1 = Dropout(0.5)(video_flow_new)
    dense1 = Dense(128, kernel_initializer='normal', activation='relu')(drop1)

    drop2 = Dropout(0.5)(dense1)
    dense2 = Dense(4, kernel_initializer='normal')(drop2)
    out = Activation('softmax')(dense2)
    model = Model(inputs=[inputs], outputs=out)
    sgd = optimizers.SGD(lr=0.0001, decay=0.00001, momentum=0.9)
    adam = optimizers.Adam(lr=0.001, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    filepath = "weights_microexpstcnn/CASME_LOSOsub" + str(sub) + "_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2)
    callbacks_list = [checkpoint]
    numpy.save("numpy_validation_dataset/CASME_LOSOsub" + str(sub) + "_val_images.npy", testing_set)
    numpy.save("numpy_validation_dataset/CASME_LOSOsub" + str(sub) + "val_labels.npy", testing_labels)
    print(training_set.shape)
    print(testing_set.shape)
    print(training_labels.shape)
    print(testing_labels.shape)
    hist = model.fit(training_set, training_labels, validation_data=(testing_set, testing_labels),
                     callbacks=callbacks_list, batch_size=8, nb_epoch=100)
    # predict
    # print("predict subject" + str(sub))
    model = load_model(filepath)
    predict = model.predict(testing_set, batch_size=8)

    # print(predict)
    print("cm subject" + str(sub))
    testing_labels = numpy.argmax(testing_labels, axis=1)
    predict = numpy.argmax(predict, axis=1)
    ct = confusion_matrix(testing_labels, predict)
    print(ct)
    # check the order of the CT
    order = numpy.unique(numpy.concatenate((predict, testing_labels)))
    # create an array to hold the CT for each CV
    mat = numpy.zeros((4, 4))

    # put the order accordingly, in order to form the overall ConfusionMat
    for m in range(len(order)):
        for n in range(len(order)):
            mat[int(order[m]), int(order[n])] = ct[m, n]

    tot_mat = mat + tot_mat
    print(tot_mat)
    if sub == 26:
        # microAcc = numpy.trace(tot_mat) / numpy.sum(tot_mat)
        [f1, precision, recall] = fpr(tot_mat, 4)
        war = weighted_average_recall(tot_mat, 4, 246)
        uar = unweighted_average_recall(tot_mat, 4)
        print("f1:" + str(f1))
        print("war: " + str(war))
        print("uar: " + str(uar))
        print("cm:")
        print(tot_mat)



