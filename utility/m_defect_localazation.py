import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2, scipy


from keras.models import Model
from matplotlib.patches import Rectangle
from skimage.feature.peak import peak_local_max
from keras.models import load_model

# Load the prediction model
model = load_model('models/our_model_with_GAP.h5')
print(model.summary())
# Ask the user for the input image path
image_path = input("Enter the path to the input image: ")

# Read the input image
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = image.astype('float32') / 255.0
image = cv2.resize(image, (112, 112))
img_predected = np.argmax(model.predict(np.expand_dims(image, axis=0)))

defected_image_idx = np.where((img_predected == 0)|(img_predected == 2))[0]

#capture it in memory as an array
predicted_as_def=[]
for i in defected_image_idx:
    def_img = image
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
    last_conv_model = Model(model.input, model.get_layer("conv2d_12").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    
    #Upsample/resize the last conv. output to same size as original image
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 132)), last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    
    #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    #with rel threshold of 0.5 (compared to the max peak). 
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=5) 

    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3))
    plt.imshow(heat_map, cmap='jet', alpha=0.20)
    plt.show()
    for i in range(0,peak_coords.shape[0]):
        print(i)
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-25, y-25), 50,50,linewidth=1,edgecolor='r',facecolor='none'))


heat_map = plot_heatmap(predicted_as_def[0])
img = predicted_as_def[0]
plt.imshow(predicted_as_def[0])
plt.show()