
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, os
import seaborn as sns
import tensorflow as tf

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

tf.random.set_seed(16)

print(os.listdir("images/"))

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

#######################################
# building model

activation = 'relu'

feature_extractor = Sequential()

# Block 1

feature_extractor.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=activation, input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 2

feature_extractor.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 3

feature_extractor.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 4

feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block 5

feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(Conv2D(132, kernel_size=(3, 3), padding="same", activation=activation))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPool2D((2, 2), strides=(2, 2)))

feature_extractor.add(Flatten())

#Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(x_train) #This is out X input to RF

# Range of values for number of estimators
estimator_range = range(1, 151)

# List to store accuracy values
accuracy_scores = []

# Iterate over different number of estimators
for n_estimators in estimator_range:
    # Create the RandomForestClassifier
    RF_model = RandomForestClassifier(n_estimators = n_estimators, random_state=367)

    #Train the model on training data
    RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding

    #Send test data through same feature extractor process
    X_test_feature = feature_extractor.predict(x_test)
    #Now predict using the trained RF model. 
    prediction_RF = RF_model.predict(X_test_feature)
    #Inverse le transform to get original label back. 
    #prediction_RF = le.inverse_transform(prediction_RF)
    # Calculate and store the accuracy
    accuracy = metrics.accuracy_score(y_test, prediction_RF)
    accuracy_scores.append(accuracy)

# Plot the accuracy vs. number of estimators
plt.plot(estimator_range, accuracy_scores, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Estimators')
plt.grid(True)
plt.show()



#Print overall accuracy
#test_labels = le.inverse_transform(y_test)
#print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

#Confusion Matrix - verify accuracy of each class

#cm = confusion_matrix(test_labels, prediction_RF)
#print(cm)
#sns.heatmap(cm, annot=True)
#plt.show()


#Check results on a few select images
#n=9 #Select the index of image to be loaded for testing
#img = x_test[n]
#plt.imshow(img)
#input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
#input_img_features=feature_extractor.predict(input_img)
#prediction_RF = RF_model.predict(input_img_features)[0] 
#prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
#print("The prediction for this image is: ", prediction_RF)
#print("The actual label for this image is: ", test_labels[n])