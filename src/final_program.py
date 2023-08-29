import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from matplotlib.patches import Rectangle
from skimage.feature.peak import peak_local_max
import scipy

# Load the prediction model for piece classification
piece_classification_model = load_model('models/cnn_model.h5')

# Load the heatmap generation model
heatmap_model = load_model('models/pre_trained_model.h5')

# Ask the user for the input image path
image_path = input("Enter the path to the input image: ")

# Read the input image
image = cv2.imread(image_path)
image_for_heatmap = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the dimensions of each smaller piece
piece_height = height // 1
piece_width = width // 1

# Initialize an empty list to store the selected defective pieces
selected_defective_pieces = []

# Divide the image into pieces
for i in range(1):
    for j in range(1):
        piece_hsv = hsv_image[i*piece_height:(i+1)*piece_height, j*piece_width:(j+1)*piece_width]
        
        # Create a mask based on the HSV color range
        lower = np.array([0, 0, 85])
        upper = np.array([179, 70, 255])
        mask = cv2.inRange(piece_hsv, lower, upper)
        
        white_pixels = np.sum(mask == 255) / (piece_height * piece_width)
        
        if white_pixels >= 0.95:  # Keep only if more than 95% white pixels
            selected_defective_pieces.append(piece_hsv)

# Process and generate heatmap for each defective piece
for idx, defective_piece_hsv in enumerate(selected_defective_pieces):
    
    # Prepare the piece for classification (normalize)
    defective_piece_hsv = cv2.cvtColor(defective_piece_hsv, cv2.COLOR_HSV2BGR)
    piece_for_class = defective_piece_hsv.astype('float32') / 255.0
    
    #Resize selected pieces to 112x112
    piece_for_class = cv2.resize(piece_for_class, (112, 112))

    # Classify the piece
    piece_prediction = piece_classification_model.predict(np.expand_dims(piece_for_class, axis=0))
    piece_class = np.argmax(piece_prediction)
    
    # Get the probabilities for each class
    probabilities = piece_prediction[0]
    
    # Display the piece with probability distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.resize(defective_piece_hsv, (224, 224)))
    
    # If the piece is classified as defective
    if piece_class != 1:
        plt.title(f'Defective segment {idx}', fontsize=14)
        piece_for_class = cv2.resize(piece_for_class, (224, 224))
        # Generate heatmap
        pred = heatmap_model.predict(np.expand_dims(piece_for_class, axis=0))
        pred_class = np.argmax(pred)
        
        last_layer_weights = heatmap_model.layers[-1].get_weights()[0]
        
        last_layer_weights_for_pred = last_layer_weights[:, pred_class]
        
        last_conv_model = Model(heatmap_model.input, heatmap_model.get_layer("block5_conv3").output)
        last_conv_output = last_conv_model.predict(piece_for_class[np.newaxis,:,:,:])
        last_conv_output = np.squeeze(last_conv_output)
        h = int(piece_for_class.shape[0] / last_conv_output.shape[0])
        w = int(piece_for_class.shape[1] / last_conv_output.shape[1])
        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
        heat_map = np.dot(upsampled_last_conv_output.reshape((piece_for_class.shape[0]*piece_for_class.shape[1], 512)), 
                         last_layer_weights_for_pred).reshape(piece_for_class.shape[0], piece_for_class.shape[1])
        #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
        #with rel threshold of 0.5 (compared to the max peak). 
        peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=5) 
        #plt.imshow(image_for_heatmap, cmap='gray', alpha=0.3)
        plt.imshow(heat_map, cmap='jet', alpha=0.4)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f'Defective segment {idx}', fontsize=14)
        plt.imshow(piece_for_class)
        plt.axis('off')
        for i in range(0,peak_coords.shape[0]):
            y = peak_coords[i,0]
            x = peak_coords[i,1]
            plt.gca().add_patch(Rectangle((x-25, y-25), 50,50,linewidth=1,edgecolor='r',facecolor='none'))

    else:
        plt.title(f'Good segment {idx}', fontsize=14)
    
    # Display the class probabilities as horizontal bar chart
    plt.subplot(1, 3, 3)
    class_names = {0: 'crack', 1: 'good', 2: 'pinhole'}
    plt.barh(list(class_names.values()), probabilities, color='blue', height=0.5)
    plt.xlabel('Probability')
    plt.title('Class Probabilities', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the probability values at the end of the bars
    for i, prob in enumerate(probabilities):
        plt.text(prob + 0.01, i, f'{prob:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()