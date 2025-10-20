import numpy as np

def euclidean_distance(image_one, image_two):
    """
    calculate the euclidean distance between two images
    """
    return np.sqrt(np.sum((image_one - image_two)**2))


def prediction(x_training_input, y_training_output, test_image, k):
    """
    predict the label of an image using KNN
    """
    # calculate the distance from the image we are currently analyzing to every other image
    distances = []
    for current_training_image in x_training_input:
        current_distance = euclidean_distance(test_image, current_training_image)
        distances.append(current_distance)
    
    # get the sorted indicies of the distances
    sorted_distance_indices = np.argsort(distances)
    # from those, we only want the k nearest ones
    k_nearest_indices = sorted_distance_indices[:k]
    
    # get the labels of the k nearest neighbors
    k_nearest_labels = []
    for i in k_nearest_indices:
        label = y_training_output[i]
        k_nearest_labels.append(label)

    vote_tally = np.bincount(k_nearest_labels) # tally the votes
    winning_label = np.argmax(vote_tally) # pick the label with the most votes
        
    return winning_label
    

import data
import time

if __name__ == '__main__':
    print("Starting the KNN Classifier...")
    # load the data
    x_training_input, y_training_output, x_testing_input, y_testing_output = data.preprocess_data()

    print(f"\nCalculating...")
    test_samples = 100 # we will test on 100 samples since KNN is slow
    test_images = x_testing_input[:test_samples]
    test_labels = y_testing_output[:test_samples]

    # test the sample at each k value
    test_ks = [1, 3, 5]
    for k_value in test_ks:
        print(f"\n [!] Testing at k = {k_value}")
        correct_predictions = 0
        start_time = time.time()

        # go through each of the test samples
        for i in range(test_samples):
            current_image = test_images[i]
            actual_label = test_labels[i]

            # get the prediction labels
            predicted_label = prediction(x_training_input, y_training_output, current_image, k_value)

            # check if our prediction was correct
            if predicted_label == actual_label:
                correct_predictions += 1
            
            # TESTING show the progress as we go through each image
            # print(f"Testing image {i + 1}/{test_samples}...", end='\r')

        # calculate the accuracy
        accuracy = (correct_predictions / test_samples) * 100 # to get a pretty percent value
        end_time = time.time() # end the timer for the current k
        duration = end_time - start_time # calculate the elapsed time

        print(f"\n******************************************************")
        print(f"Test complete for k = {k_value}!")
        print(f"The accuracy is {accuracy:.1f}% ---> {correct_predictions}/{test_samples} correct predictions!")
        print(f"The test took {duration:.1f} seconds!")
        print(f"******************************************************")
        
"""OUTPUT
Calculating...

 [!] Testing at k = 1

******************************************************
Test complete for k = 1!
The accuracy is 97.0% ---> 97/100 correct predictions!
The test took 18.6 seconds!
******************************************************

 [!] Testing at k = 3

******************************************************
Test complete for k = 3!
The accuracy is 98.0% ---> 98/100 correct predictions!
The test took 18.5 seconds!
******************************************************

 [!] Testing at k = 5

******************************************************
Test complete for k = 5!
The accuracy is 98.0% ---> 98/100 correct predictions!
The test took 18.9 seconds!
******************************************************
        
"""