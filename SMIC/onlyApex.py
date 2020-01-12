
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape,Permute
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import dot,concatenate,Lambda,RepeatVector
import numpy
import tensorflow as tf
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import keras.optimizers as optimizers
from keras.optimizers import SGD
K.set_image_dim_ordering('th')
from keras import optimizers

image_rows, image_columns, image_depth = 160, 160, 2
videoopath='../../datasets/SMIC_FIRST_APEX/'

tot_mat = numpy.zeros((3, 3))
for sub in range(1,21):
    print(".starting subject" + str(sub))
    subname='s'+str(sub)+'_'
    training_list = []
    testing_list=[]
    training_labels = []
    testing_labels = []
   
    emotionlisting = os.listdir(videoopath)
    for emotiondata in emotionlisting:#neg,pos,sur
        negativepath = videoopath + emotiondata + "/"
        directorylisting = os.listdir(negativepath)
        i = 0
        for video in directorylisting:
            i = i + 1
            videopath = negativepath + video 
            frames = []
            framelisting = os.listdir(videopath) 
            framerange = [x for x in range(2)]
            for frame in framerange:
                imagepath = videopath + "/" + framelisting[frame+2]
                image = cv2.imread(imagepath)
                grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frames.append(grayimage)
            videoarray = numpy.asarray(frames)                     
            if subname in video:
                testing_list.append(videoarray)
                if 'Negative' in emotiondata:
                    testing_labels.append(0)
                elif 'Positive'in emotiondata:
                    testing_labels.append(1)
                else:
                    testing_labels.append(2)

            else:
                training_list.append(videoarray)
                if 'Negative' in emotiondata:
                    training_labels.append(0)
                   
                elif 'Positive'in emotiondata:
                    training_labels.append(1)
                   
                else:
                    training_labels.append(2)
                   

    # -------------------------------------------------------------------------------------------
    training_list = numpy.asarray(training_list)
    trainingsamples = len(training_list)
    testing_list=numpy.asarray(testing_list)
    testingsamples = len(testing_list)
    if testingsamples==0:
        continue    
    testing_labels=numpy.asarray(testing_labels)
    training_labels = numpy.asarray(training_labels)
    training_labels = np_utils.to_categorical(training_labels, 3)
    testing_labels = np_utils.to_categorical(testing_labels, 3)

    training_set = numpy.zeros((trainingsamples, 1,image_depth, image_rows, image_columns ))
    testing_set = numpy.zeros((testingsamples, 1, image_depth,image_rows, image_columns ))
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
   
    inputs = Input(shape=(1, image_depth,image_rows, image_columns ))#4,160,160
   
    #conv-pooling
    videoconv1=Convolution3D(32,(1,3,3,),activation='relu')(inputs)#3,158,158
    videomax1=MaxPooling3D(pool_size=(1,3,3))(videoconv1)#3,52,52
    videoconv2=Convolution3D(32,(1,3,3),activation='relu')(videomax1)#2,50,50
    videomax2=MaxPooling3D(pool_size=(1,3, 3))(videoconv2)#4,16,16

    #self-attention
   
    video_data_attention = Reshape(target_shape=(32,512))(videomax2)
    video_att_1=Activation('tanh')(video_data_attention)
    video_att_1 = Dense(1)(video_att_1)    
    video_att_1=Permute((2,1),input_shape=(32,1))(video_att_1)
    video_att_1 = Activation('softmax')(video_att_1)
    videodata=dot([video_att_1,video_data_attention],axes=[-1,1])#1,1024
    videodata=Flatten()(videodata)
    # dense-softmax
  
    drop1 = Dropout(0.5)(videodata)
    dense1 = Dense(64, kernel_initializer='normal', activation='relu')(drop1) 
    drop2 = Dropout(0.5)(dense1)
    dense2 = Dense(3, kernel_initializer='normal', activation='relu')(drop2)  # (3)
    out = Activation('softmax')(dense2)
    model = Model(inputs=[inputs], outputs=out)
    sgd = SGD(lr=0.0001,decay=0.00001,momentum=0.9)
    adam = optimizers.Adam(lr=0.001, decay=0.0001)
   
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    
    filepath = "weights_microexpstcnn/SMIC_LOSOsub"+str(sub)+"_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2)
    callbacks_list = [checkpoint]
    numpy.save("numpy_training_datasets/SMIC_LOSOsub"+str(sub)+"_val_images.npy", training_set)
    numpy.save("numpy_training_datasets/SMIC_LOSOsub"+str(sub)+"val_labels.npy", training_labels)
    numpy.save("numpy_validation_dataset/SMIC_LOSOsub"+str(sub)+"_val_images.npy", testing_set)
    numpy.save("numpy_validation_dataset/SMIC_LOSOsub"+str(sub)+"val_labels.npy", testing_labels)
    hist = model.fit(training_set, training_labels, validation_data=(testing_set, testing_labels),
                     callbacks=callbacks_list, batch_size=8, nb_epoch=100, shuffle=True)
    #predict
    #print("predict subject" + str(sub))
    model=load_model(filepath,compile=False)
    predict=model.predict(testing_set, batch_size=16)
    #print(predict)
    print("cm subject"+str(sub))
    testing_labels=numpy.argmax(testing_labels,axis=1)
    predict=numpy.argmax(predict,axis=1)
    ct = confusion_matrix(testing_labels,predict)
    print(ct)
    # check the order of the CT
    order = numpy.unique(numpy.concatenate((predict,testing_labels)))
		# create an array to hold the CT for each CV
    mat = numpy.zeros((3,3))
    
		# put the order accordingly, in order to form the overall ConfusionMat
    for m in range(len(order)):
      for n in range(len(order)):
        mat[int(order[m]),int(order[n])]=ct[m,n]
			   
    tot_mat = mat + tot_mat
    print(tot_mat)
    if sub ==20:
    #microAcc = numpy.trace(tot_mat) / numpy.sum(tot_mat)
      [f1,precision,recall] = fpr(tot_mat,3)
      war = weighted_average_recall(tot_mat, 3, 164)
      uar = unweighted_average_recall(tot_mat, 3)
      print("f1:"+str(f1))		
      print("war: " + str(war))
      print("uar: " + str(uar))
      print("cm:")
      print(tot_mat)
    
   		

