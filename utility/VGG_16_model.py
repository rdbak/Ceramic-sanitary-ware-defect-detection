
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, os
import seaborn as sns
import tensorflow as tf

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Set a seed for the weights initialization
tf.random.set_seed(16)

print(os.listdir("images/"))

SIZE = 224

train_images = []
train_labels = []
for directory_path in glob.glob("images/*"):
    label = directory_path.split("/")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

input_images = np.array(train_images)
input_labels = np.array(train_labels)

#Encode Labels from text to integers
le = preprocessing.LabelEncoder()
le.fit(input_labels)
input_labels_encoded = le.transform(input_labels)

# Scaling pixel values to between 0 and 1
input_images = input_images/255.0

#Split data into test and train datasets (already slpit in our case but assigning to meaninful variables)
x_train, x_test, y_train, y_test = train_test_split(input_images, input_labels_encoded, test_size=0.2, random_state=25)

##########################################################################################################
#One hot encode y values for neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#######################################
# building model

activation = 'relu'

feature_extractor = Sequential()

# Block 1

feature_extractor.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=activation, input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))
#feature_extractor.add(Dropout(0.1))


# Block 2

feature_extractor.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 3

feature_extractor.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))
#feature_extractor.add(Dropout(0.1))

# Block 4

feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 5

feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

feature_extractor.add(Flatten())

#Adding Layers for deep Learning prediction
x = feature_extractor.output
x = Dense(4096, activation = activation)(x)
x = Dense(4096, activation = activation)(x)
predection_layer = Dense(3, activation='softmax')(x)

#Make a new model combining feature extractor and DL
cnn_model = Model(inputs=feature_extractor.input, outputs=predection_layer)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary())

###############################################
#Train the CNN model
history = cnn_model.fit(x_train, y_train_one_hot, epochs=15, validation_data = (x_test, y_test_one_hot))

#plot the training and validation accuracy and loss at each epoch

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
test_labels = le.inverse_transform(y_test)
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)
plt.show()


#Check the result in a few selected images

n=15 #Select the index of image to be Loaded for testing
img = x_test[n]
input_img = np.expand_dims(img, axis=0) #Expand dims so the inputs is (num images, x, y, c)
prediction = np.argmax(cnn_model.predict(input_img)) #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])
plt.imshow(img)
plt.show()