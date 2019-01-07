
# coding: utf-8

# In[10]:

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D,Activation,Dropout,Cropping2D
from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split


# In[11]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[63]:

samples = []
sample_paths = ['./data/driving_log.csv','./data/driving_log_1.csv','./data/driving_log_2.csv']

for sample_path in sample_paths:
    with open("./data/driving_log.csv") as csv_file:
        flag = 1
        dump = csv.reader(csv_file)
        for line in dump:
            if flag == 1:
                flag = 0
                continue
            samples.append(line)

left = []
right =[]
center = []
for s in samples:
    angle=float(s[3])
    if  angle>0.1:
        right.append(s)
    elif angle<-0.1:
        left.append(s)
    else:
        center.append(s)

print("Initial sample size : "+str(len(samples)))
print("Left angle : "+str(len(left)))
print("Center angle : "+str(len(center)))
print("Right angle : "+str(len(right)))

ac_data = [len(right),len(left),len(center)]
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Before Processing')
plt.bar(range(3),ac_data,align='center')

center = center[::4] 
plt.subplot(1,2,2)
plt.title('After Processing')
plt.bar(range(3),[len(right),len(left),len(center)])
plt.show()

samples = []
samples.extend(left)
samples.extend(right)
samples.extend(center)

train_samples, validation_samples = train_test_split(samples, test_size=0.196)
print("Traing set size : "+str(len(train_samples)))

###############################################

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=22)
validation_generator = generator(validation_samples, batch_size=22)

###############################################


# In[64]:

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(4, 5, 5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8, 5, 5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(16, 3, 3,activation='relu'))
model.add(Flatten())

model.add(Dense(800,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='relu'))

model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
callbacks_list = [ModelCheckpoint('model.h5', save_best_only=True)]
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validation_generator,callbacks=callbacks_list, nb_val_samples=len(validation_samples), nb_epoch=3)
model.summary()


# In[ ]:



