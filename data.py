import os
import numpy as np
from PIL import Image #to process images

def load_data(folderPath):
    """
    load the MNIST data from images
    """
    
    images = []
    labels = []

    # go through each digit's folder (0-9)
    for current_digit in range(10):
        print(f"Loading images for folder: {current_digit}.")
        digit_path = os.path.join(folderPath, str(current_digit))
        
        # check if the folder exists
        if not os.path.isdir(digit_path):
            continue # if it doesn't exist, skip it
        
        # go through each image in the folder
        for file_name in os.listdir(digit_path): 
            file_path = os.path.join(digit_path, file_name) # make the full file path
        
            # convert image to grey-scale
            img = Image.open(file_path).convert('L')
            
            # convert image to Numpy array
            img_array = np.array(img)
            
            # add data
            images.append(img_array)
            
            # add label
            labels.append(current_digit)


    # Convert the lists of images and labels to NumPy arrays
    return (np.array(images), np.array(labels))


def preprocess_data():
    # load the data from the file
    images, labels = load_data(r'D:\AI_Project1\DATA\MNIST')
    
    # check if the images loaded and show the details
    if len(images) > 0:
        print(f"\nWe loaded {len(images)} images.")
        print(f"The first image is {images[0].shape}")
        print(f"The image array is {images.shape}")
        print(f"The labels array is {labels.shape}")
    else:
        print("Failed to  load images. Check the path.")
        
    # normalize the data (we divide by 255 to get a value between 0 t 1)
    print(f"\nNormalizing the data...")
    normalized_images = images / 255.0
    print(f"After normalizing we get {normalized_images.shape}")
    
    number_of_images = len(normalized_images)
    
    # flatten the data (we may need 1d data so we flatten 28x28 to 784)
    flattened_images = normalized_images.reshape(number_of_images, 784)
    print(f"After flattening we get {flattened_images.shape}")
    
    print(f"\nShuffling the data...")
    # shuffle the data so we get random samples
    index_of_image = np.arange(number_of_images)
    np.random.shuffle(index_of_image)
    
    shuffled_images = flattened_images[index_of_image]
    shuffled_labels = labels[index_of_image]
    
    print(f"\nSplitting the data...")
    # we need to split the data between training and testing
    split_point = int(0.8 * number_of_images) # 80% testing | 20% training -- general purpose
    """
    split_point = int(0.75 * number_of_images) # 75% testing | 25% training -- when we need a bigger test set
    split_point = int(0.70 * number_of_images) # 70% testing | 30% training -- when we have a smaller data set 
    """
    
    # x is our input
    x_training_input = shuffled_images[:split_point]
    # y is our expected output
    y_training_output = shuffled_labels[:split_point]
    
    x_testing_input = shuffled_images[split_point:]
    y_testing_output = shuffled_labels[split_point:]
    
    return x_training_input, y_training_output, x_testing_input, y_testing_output

# for a cnn we use 2D images, so we will use the same preprocess w/o flattening
def preprocess_cnn():
    # load the data from the file
    images, labels = load_data(r'D:\AI_Project1\DATA\MNIST')
    
    # check if the images loaded and show the details
    if len(images) > 0:
        print(f"\nWe loaded {len(images)} images.")
        print(f"The first image is {images[0].shape}")
        print(f"The image array is {images.shape}")
        print(f"The labels array is {labels.shape}")
    else:
        print("Failed to  load images. Check the path.")
        
    # normalize the data (we divide by 255 to get a value between 0 t 1)
    print(f"\nNormalizing the data...")
    normalized_images = images / 255.0
    print(f"After normalizing we get {normalized_images.shape}")
    
    number_of_images = len(normalized_images)
    
    print(f"\nShuffling the data...")
    # shuffle the data so we get random samples
    index_of_image = np.arange(number_of_images)
    np.random.shuffle(index_of_image)
    
    shuffled_images = normalized_images[index_of_image]
    shuffled_labels = labels[index_of_image]
    
    print(f"\nSplitting the data...")
    # we need to split the data between training and testing
    split_point = int(0.8 * number_of_images) # 80% testing | 20% training -- general purpose
    """
    split_point = int(0.75 * number_of_images) # 75% testing | 25% training -- when we need a bigger test set
    split_point = int(0.70 * number_of_images) # 70% testing | 30% training -- when we have a smaller data set 
    """
    
    # x is our input
    x_training_input = shuffled_images[:split_point]
    # y is our expected output
    y_training_output = shuffled_labels[:split_point]
    
    x_testing_input = shuffled_images[split_point:]
    y_testing_output = shuffled_labels[split_point:]
    
    return x_training_input, y_training_output, x_testing_input, y_testing_output  
    

# TESTING
if __name__ == '__main__':
    # get the data
    x_training_input, y_training_output, x_testing_input, y_testing_output = preprocess_data()
    
    print(f"\nx-training input shape: {x_training_input.shape}")
    print(f"y-training output shape: {y_training_output.shape}")
    print(f"\nx-testing input shape: {x_testing_input.shape}")
    print(f"y-testing output shape: {y_testing_output.shape}")
    
"""OUTPUT
Loading images for folder: 0.
Loading images for folder: 1.
Loading images for folder: 2.
Loading images for folder: 3.
Loading images for folder: 4.
Loading images for folder: 5.
Loading images for folder: 6.
Loading images for folder: 7.
Loading images for folder: 8.
Loading images for folder: 9.

We loaded 60000 images.
The first image is (28, 28)
The image array is (60000, 28, 28)
The labels array is (60000,)

Normalizing the data...
After normalizing we get (60000, 28, 28)
After flattening we get (60000, 784)

Shuffling the data...

Splitting the data...

x-training input shape: (48000, 784)
y-training output shape: (48000,)

x-testing input shape: (12000, 784)
y-testing output shape: (12000,)
"""