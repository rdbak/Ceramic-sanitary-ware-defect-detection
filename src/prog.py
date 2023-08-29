import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the prediction model
model = load_model('models/cnn_model.h5')

# Ask the user for the input image path
image_path = input("Enter the path to the input image: ")

# Read the input image
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the dimensions of each smaller piece
piece_height = height // 5
piece_width = width // 5

# Initialize an empty list to store the selected pieces
selected_pieces = []

# Divide the image into pieces
for i in range(5):
    for j in range(5):
        piece_hsv = hsv_image[i*piece_height:(i+1)*piece_height, j*piece_width:(j+1)*piece_width]
        
        # Create a mask based on the HSV color range
        lower = np.array([0, 0, 85])
        upper = np.array([179, 70, 255])
        mask = cv2.inRange(piece_hsv, lower, upper)
        
        white_pixels = np.sum(mask == 255) / (piece_height * piece_width)
        
        if white_pixels >= 0.95:  # Keep only if more than 95% white pixels
            selected_pieces.append(piece_hsv)

# Resize selected pieces to 112x112
resized_pieces = [cv2.resize(piece, (112, 112)) for piece in selected_pieces]

# Display and save the resized pieces
for idx, piece_hsv in enumerate(resized_pieces):
    plt.figure(figsize=(10, 5))
    
    # Prepare the piece for prediction (normalize)
    piece_hsv = cv2.cvtColor(piece_hsv, cv2.COLOR_HSV2BGR)
    piece_for_pred = piece_hsv.astype('float32') / 255.0
    
    # Make the prediction
    prediction = model.predict(np.expand_dims(piece_for_pred, axis=0))
    
    # Map the numerical class labels to class names
    class_names = {0: 'crack', 1: 'good', 2: 'pinhole'}
    
    # Get the predicted class label
    predicted_label = np.argmax(prediction)
    
    # Get the predicted class name
    predicted_class = class_names[predicted_label]
    
    # Get the probabilities for each class
    probabilities = prediction[0]
    
    # Display the piece
    plt.subplot(1, 2, 1)
    plt.imshow(piece_hsv)
    plt.title(f'Piece {idx} Prediction: {predicted_class}', fontsize=14)
    plt.axis('off')
    
    # Display the class probabilities as a horizontal bar chart
    plt.subplot(1, 2, 2)
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