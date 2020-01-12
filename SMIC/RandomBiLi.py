
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape,Permute
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.optimizers import SGD
from keras.layers import dot,concatenate,Lambda,RepeatVector
import numpy
import keras.optimizers as optimizers
from sklearn.metrics import confusion_matrix
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
K.set_image_dim_ordering('th')
# limit memory


image_rows, image_columns, image_depth = 64, 64, 26
videoopath='../../datasets/SMIC_DETAIL_TIM10/'
flowpath='../../datasets/SMIC_DEEPFLOW10/'
training_list = []
emotionlisting=os.listdir(videoopath)
for emotiondata in emotionlisting:
    negativepath=videoopath+emotiondata+"/"
    negflowepath=flowpath+emotiondata+"/"
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
        training_list.append(videoarray)
#-------------------------------------------------------------------------------------------
training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)
traininglabels = numpy.zeros((trainingsamples, ), dtype = int)
traininglabels[0:70] = 0
traininglabels[70:121] = 1
traininglabels[121:164] = 2
traininglabels = np_utils.to_categorical(traininglabels, 3)
training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)


inputs = Input(shape=(1,image_rows, image_columns, image_depth))

videorow = Lambda(lambda x: x[:,:,:,:,0:10])(inputs)
flowrow=Lambda(lambda x: x[:,:,:,:,10:26])(inputs)
flowconv1=Convolution3D(32,(3,3,8),activation='relu')(flowrow)
flowmax1=MaxPooling3D(pool_size=(3, 3, 3))(flowconv1)
videoconv1=Convolution3D(32,(3,3,5),activation='relu')(videorow)
videomax1=MaxPooling3D(pool_size=(3, 3, 2))(videoconv1)
#self-attention
flow_data_attention = Reshape(target_shape=(32,1200))(flowmax1)
flow_att_1=Activation('tanh')(flow_data_attention)
flow_att_1 = Dense(3)(flow_att_1)#(64,32)    
flow_att_1= Permute((2,1),input_shape=(32,3))(flow_att_1)
flow_att_1 = Activation('softmax')(flow_att_1)
flowdata=dot([flow_att_1,flow_data_attention],axes=[-1,1])#1,1200
flowdata=Reshape(target_shape=(9,400))(flowdata)
    
video_data_attention = Reshape(target_shape=(32,1200))(videomax1)
video_att_1=Activation('tanh')(video_data_attention)
video_att_1 = Dense(3)(video_att_1)    
video_att_1=Permute((2,1),input_shape=(32,3))(video_att_1)
video_att_1 = Activation('softmax')(video_att_1)
videodata=dot([video_att_1,video_data_attention],axes=[-1,1])
videodata=Reshape(target_shape=(9,400))(videodata)            


#flow<->video
att_1 = concatenate([ videodata,flowdata], axis=-1)
att_1=Reshape(target_shape=(9,800))(att_1)
att_1 = Activation('tanh')(att_1)
att_1 = Dense(1)(att_1)
att_1 = Activation('softmax')(att_1)
att_1_pro = Flatten()(att_1)
flow_new=dot([att_1_pro,flowdata],axes=(1,1))

flow_new_repeat=RepeatVector(9)(flow_new)
att_2 = concatenate([flow_new_repeat, videodata], axis=-1)
att_2=Reshape(target_shape=(9,800))(att_2)
att_2 = Activation('tanh')(att_2)
att_2 = Dense(1)(att_2)
att_2 = Activation('softmax')(att_2)
att_2_pro = Flatten()(att_2)
video_new=dot([att_2_pro,videodata],axes=(1,1))

video_flow_new = concatenate([video_new, flow_new],axis=-1)
dense1=Dense(128, kernel_initializer='normal', activation='relu')(video_flow_new)
drop2=Dropout(0.5)(dense1)
dense2=Dense(3, kernel_initializer='normal')(drop2)
out=Activation('softmax')(dense2)
model = Model(inputs=[inputs], outputs=out)
sgd = SGD(lr=0.009,decay=0.0009,momentum=0.9)
adam = optimizers.Adam(lr=0.001, decay=0.0001)
model.compile(loss = 'categorical_crossentropy', optimizer =adam, metrics = ['accuracy'])
print(model.summary())
filepath="weights_microexpstcnn/SMIC_RandomSuiji_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)
numpy.save('numpy_validation_dataset/SMIC_RandomSuiji_val_images.npy', validation_images)
numpy.save('numpy_validation_dataset/SMIC_RandomSuiji_val_labels.npy', validation_labels)
hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)


model=load_model(filepath,compile=False)
predict=model.predict(validation_images, batch_size=16)
predict=numpy.argmax(predict,axis=1)
testing_labels=numpy.argmax(validation_labels,axis=1)
num = len(testing_labels)
ct = confusion_matrix(testing_labels,predict)
print(ct)
order = numpy.unique(numpy.concatenate((predict,testing_labels)))
mat = numpy.zeros((3,3))
for m in range(len(order)):
  for n in range(len(order)):
    mat[int(order[m]),int(order[n])]=ct[m,n]
[f1,precision,recall] = fpr(mat,3)
war = weighted_average_recall(mat, 3, num)
uar = unweighted_average_recall(mat, 3)
print("f1:"+str(f1))		
print("war: " + str(war))
print("uar: " + str(uar))

