import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt 

from keras.layers import Conv2D, MaxPooling2D

img = cv2.imread("input_image.jpg!!!")

img = cv2.resize(img, (224, 224))

model = keras.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

"""   #each time add a block to see the changes made in each block
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
"""

model.build()
model.summary()

result = model.predict(np.array([img]))


for i in range(64):
    feature_img = result[0, :, :, i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img)
plt.show()