import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, os
import seaborn as sns
import scipy
#import tensorflow as tf

from keras.optimizers import SGD
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from matplotlib.patches import Rectangle
from skimage.feature.peak import peak_local_max


#tf.random.set_seed(16)

print(os.listdir("images/"))

SIZE = 224

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#############################
#Define the model. 
#Here, we use pre-trained VGG16 layers and add GlobalAveragePooling and dense prediction layers.
#You can define any model. 
#Also, here we set the first few convolutional blocks as non-trainable and only train the last block.
#This is just to speed up the training. You can train all layers if you want. 
def get_model(input_shape = (224,224,3)):
    
    vgg = VGG16(weights='imagenet', include_top=False, input_shape = input_shape)

    #for layer in vgg.layers[:-8]:  #Set block4 and block5 to be trainable. 
    for layer in vgg.layers[:-5]:    #Set block5 trainable, all others as non-trainable
        print(layer.name)
        layer.trainable = False #All others as non-trainable.

    x = vgg.output
    x = GlobalAveragePooling2D()(x) #Use GlobalAveragePooling and NOT flatten. 
    x = Dense(3, activation="softmax")(x)  #We are defining this as multiclass problem. 

    model = Model(vgg.input, x)
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    
    return model

model = get_model(input_shape = (224,224,3))
print(model.summary())

history = model.fit(x_train, y_train, batch_size=16, epochs=30, verbose = 1, 
                    validation_data=(x_test,y_test))

#Save the trained model
#if os.path.isfile('models/pre_trained_model.h5') is False:
#    model.save('models/pre_trained_model.h5')


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

#Check model accuracy on the test data
_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#Test on single image.
n=10  #Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
print("The prediction for this image is: ", np.argmax(model.predict(input_img)))
print("The actual label for this image is: ", np.argmax(y_test[n]))


#Confusion Matrix - verify accuracy of each class
y_pred = np.argmax(model.predict(x_test), axis=1)
cm=confusion_matrix(np.argmax(y_test, axis=1), y_pred)  
sns.heatmap(cm, annot=True)
plt.show()

#############################################################
#Save all images classified as defected so we can fetch these images
#later and plot heatmaps.
########################################################
#Identify all images classified as defected
defected_image_idx = np.where((y_pred == 0)|(y_pred == 2))[0]

#capture it in memory as an array
predicted_as_def=[]
for i in defected_image_idx:
    def_img = x_test[i]
    predicted_as_def.append(def_img)

predicted_as_def = np.array(predicted_as_def)

def plot_heatmap(img):
  
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred)
    #Get weights for all classes from the prediction layer
    last_layer_weights = model.layers[-1].get_weights()[0] #Prediction layer
    #Get weights for the predicted class.
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    #Get output from the last conv. layer
    last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    
    #Upsample/resize the last conv. output to same size as original image
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 512)), 
                 last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    
    #Since we have a lot of dark pixels where the edges may be thought of as 
    #high anomaly, let us drop all heat map values in this region to 0.
    #This is an optional step based on the image. 
    #heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0
    
    #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    #with rel threshold of 0.5 (compared to the max peak). 
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10) 

    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(heat_map, cmap='jet', alpha=0.20)
    plt.show()
    for i in range(0,peak_coords.shape[0]):
        print(i)
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-25, y-25), 50,50,linewidth=1,edgecolor='r',facecolor='none'))

import random 
im = random.randint(0,predicted_as_def.shape[0]-1)
heat_map =plot_heatmap(predicted_as_def[im])

img = predicted_as_def[im]
plt.imshow(predicted_as_def[im])
plt.show()