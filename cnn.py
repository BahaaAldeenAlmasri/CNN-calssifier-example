from keras.models import Sequential
from keras.layers import Convolution2D , MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from keras.preprocessing import image


Classifire= Sequential()
Classifire.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))
Classifire.add(MaxPooling2D(pool_size=(2,2)))
Classifire.add(Flatten())

Classifire.add(Dense(128,activation='relu'))
Classifire.add(Dense(1,activation='sigmoid'))

Classifire.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255)

training_set =train_datagen.flow_from_directory(
    'training_set',
    target_size=(64,64),
    class_mode="binary")


test_set =test_datagen.flow_from_directory(
    'test_set',
    target_size=(64,64),
    class_mode="binary")
 

Classifire.fit(
    training_set,
    steps_per_epoch=60,
    epochs=30,
    validation_data=test_set,
    validation_steps=800)


test_image= image.image_utils.load_img('dog4047.jpg',target_size=(64,64))
test_image= image.image_utils.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis=0)
result = Classifire.predict(test_image)
training_set.class_indices
if result[0][0]>=0.5:
    prediction='dog'
else: prediction="cat"

print(prediction)

