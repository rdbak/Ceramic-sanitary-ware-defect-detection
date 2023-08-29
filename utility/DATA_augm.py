"""
Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""

from keras.preprocessing.image import ImageDataGenerator
#from skimage import io

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1],
        fill_mode='nearest')    #Also try constant, reflect, wrap


#####################################################################
#Multiclass. Read dirctly from the folder structure using flow_from_directory

i = 0
for batch in datagen.flow_from_directory(directory='images folder location', 
                                         batch_size=16,  
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir='saving directory',
                                         save_prefix='aug', 
                                         save_format='jpg'):
    i += 1
    if i > 32:
        break 

#Creates 32 images for each class. 
