#!/usr/bin/env python
# coding: utf-8

# # Multi-class Weather Classification using Alexnet

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[2]:


path = r'C:\Users\HP\Downloads\imgdataset\Multi-class Weather Dataset\train'
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(227,227), class_mode='categorical')


# In[7]:


d = train.class_indices
d


# In[8]:


print("Batch Size for Input Image : ",train[0][0].shape)
print("Batch Size for Output Image : ",train[0][1].shape)
print("Image Size of first image : ",train[0][0][0].shape)
print("Output of first image : ",train[0][1][0].shape)


# In[9]:


fig , axs = plt.subplots(2,3 ,figsize = (10,10))
axs[0][0].imshow(train[0][0][12])
axs[0][0].set_title(train[0][1][12])
axs[0][1].imshow(train[0][0][10])
axs[0][1].set_title(train[0][1][10])
axs[0][2].imshow(train[0][0][5])
axs[0][2].set_title(train[0][1][5])
axs[1][0].imshow(train[0][0][20])
axs[1][0].set_title(train[0][1][20])
axs[1][1].imshow(train[0][0][25])
axs[1][1].set_title(train[0][1][25])
axs[1][2].imshow(train[0][0][3])
axs[1][2].set_title(train[0][1][3])


# In[10]:


def AlexNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(5,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(10,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(15, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(20, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(25, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(40, activation = 'relu', name = "fc0")(X)
    
    X = Dense(40, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(5,activation='softmax',name = 'fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AlexNet')
    return model


# In[11]:


alex = AlexNet(train[0][0].shape[1:])


# In[12]:


alex.summary()


# In[13]:


alex.compile(optimizer = 'SGD' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[14]:


alex.fit_generator(train,epochs=20)


# In[15]:


path_test = r'C:\Users\HP\Downloads\imgdataset\Multi-class Weather Dataset\validation'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(path_test, target_size=(227,227), class_mode='categorical')


# In[16]:


preds = alex.evaluate_generator(test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[17]:


path_test = r'C:\Users\HP\Downloads\imgdataset\Multi-class Weather Dataset\validation'
predict_datagen = ImageDataGenerator(rescale=1. / 255)
predict = predict_datagen.flow_from_directory(path_test, target_size=(227,227), batch_size = 1,class_mode='categorical')


# In[18]:


predictions = alex.predict_generator(predict)


# In[19]:


imshow(predict[700][0][0])


# In[20]:


print(predictions[700])


# In[38]:


batch_size = 10
Y_pred = alex.predict_generator(test, 1400 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm = confusion_matrix(test.classes, y_pred)
print(cm)
print('Classification Report')
print(classification_report(test.classes, y_pred))


# In[21]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = alex.to_json()
with open('alexnet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
alex.save_weights('model_alexnet.hdf5', overwrite=True)

