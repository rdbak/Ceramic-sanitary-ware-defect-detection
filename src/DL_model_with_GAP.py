import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


# Load the prediction model
model = load_model('models/cnn_model.h5')

SIZE = 112

input_images = []
input_labels = []
for directory_path in glob.glob("images/*"):
    label = directory_path.split("/")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        input_images.append(img)
        input_labels.append(label)

input_images = np.array(input_images)
input_labels = np.array(input_labels)

#Encode Labels from text to integers
le = preprocessing.LabelEncoder()
le.fit(input_labels)
input_labels_encoded = le.transform(input_labels)

# Scaling pixel values to between 0 and 1
input_images = input_images/255.0

#Split data into test and train datasets (already slpit in our case but assigning to meaninful variables)
x_train, x_test, y_train, y_test = train_test_split(input_images, input_labels_encoded, test_size=0.2, random_state=55)

##########################################################################################################
#One hot encode y values for neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)




new_model = Sequential()
for layer in model.layers[:-6]: # go through until last layer
    new_model.add(layer)

print(new_model.summary())

for layer in new_model.layers[:-4]:    #Set block5 trainable, all others as non-trainable
        layer.trainable = False #All others as non-trainable.


x = new_model.output
x = GlobalAveragePooling2D()(x) #Use GlobalAveragePooling and NOT flatten. 
x = Dense(3, activation="softmax")(x)  #We are defining this as multiclass problem. 

new_model = Model(new_model.input, x)

new_model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

print(new_model.summary())

history = new_model.fit(x_train, y_train_one_hot, batch_size=16, epochs=50, verbose = 1, validation_data=(x_test,y_test_one_hot))

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

prediction_NN = new_model.predict(x_test) #the probability distribution of the prediction
prediction_NN = np.argmax(prediction_NN, axis=-1) #the predicted class label (the class with the highets probability)
prediction_NN = le.inverse_transform(prediction_NN) #decoding the class label from 0, 1, 2 to crack, good, pinhole

#Confusion Matrix - verify accuracy of each class
test_labels = le.inverse_transform(y_test)
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)
plt.show()


#Save the trained model
if os.path.isfile('models/our_model_with_GAP.h5') is False:
    new_model.save('models/our_model_with_GAP.h5')
