import numpy as np
import data

# to calculate the probability of each pixel being on or off
def calc_probability(x_training_input, y_training_output):
    """
    to calculate the probability of each pixel
    """
    digit_set = 10 # we have 0-9, so 10 digits in our set
    feature_size = x_training_input.shape[1] # we have 784 with our 1D flattening image
    
    prior_digit_prob = np.zeros(digit_set) # we need to store the prior probabilities for each digit
    prob_ledger = np.zeros((digit_set, feature_size)) # we need to store the probabilit of seeing a feature given the digit
    
    # we will go through each digit
    for i in range(digit_set):
        locations_of_current_digit = (y_training_output == i) # get locations of the current digit
        images_from_current_set = x_training_input[locations_of_current_digit] # select the digits from the location
        
        images_in_set = len(images_from_current_set) # number of images in our current set
        total_images = len(x_training_input) # the total number of images
        
        # we calculate the prior probability by dividing set images by total images\
        prior_digit_prob[i] = (images_in_set / total_images)
        
        # calculate probability of the current digit
        on_pixel_count = images_from_current_set.sum(axis=0) # get the count of pixels that are on for each column
        pixel_prob = (on_pixel_count / images_in_set) # get the pixel probability -- we divide the number of on pixels by the number of images in the set
        smooth_probabilities = (pixel_prob + 1e-6) # smooth the probabiliies so we do not get a 0
        prob_ledger[i] = smooth_probabilities # store the value in the ledger
        
    return prior_digit_prob, prob_ledger

def naive_bayes_prediction(test_image, trained_probs):
    """
    use our calc_probability on an image
    """
    # get our trained probabilities from the previous step
    prior_digit_prob, prob_ledger = trained_probs
    
    digit_grades = [] # to store the grades(likelihood) of each digit
    
    for digit in range(10):
        log_of_prior = np.log(prior_digit_prob[digit]) # get the log of the prior probability
        
        current_digit_map = prob_ledger[digit] # get the probability map of current digit

        # calculate the log-likelihood of the current digit
        log_prob_map = np.log(current_digit_map)
        log_inverse_prob_map = np.log(1 - current_digit_map)
        on_pixel_score = test_image * log_prob_map
        off_pixel_score = (1 - test_image) * log_inverse_prob_map
        log_likelihood = np.sum(on_pixel_score + off_pixel_score)
        
        
        total_grade = (log_of_prior + log_likelihood) # calculate the total grade
        digit_grades.append(total_grade) # add the total grade to the list
        
    # find the digit with the highest grade
    final_prediction = np.argmax(digit_grades)
    return final_prediction

if __name__ == '__main__':
    print(f"Starting Naive Bayes...\n")
    # load the data
    x_training_input, y_training_output, x_testing_input, y_testing_output = data.preprocess_data()

    print(f"\nConverting the data to a binary format...")
    x_training_binary = (x_training_input > 0.5).astype(int)
    x_testing_binary = (x_testing_input > 0.5).astype(int)

    # calculate the probabilities
    print(f"\nTraining the model by calculating the probabilities...")
    prior_probs, prob_maps = calc_probability(x_training_binary, y_training_output)
    trained_model = (prior_probs, prob_maps)

    # test the model
    print(f"\nTesting the model on the test set...")
    correct_predictions = 0
    num_test_images = len(x_testing_binary)

    for i in range(num_test_images):
        current_image = x_testing_binary[i]
        actual_label = y_testing_output[i]
        
        predicted_label = naive_bayes_prediction(current_image, trained_model) # our prediction
        
        # if it matches, we need to increment the correct prediciton count
        if predicted_label == actual_label:
            correct_predictions += 1
    
    # print the results
    accuracy = (correct_predictions / num_test_images) * 100
    print(f"\nNaive Bayes calculation complete!")
    print(f"The accuracy is {accuracy:.2f}% --> {correct_predictions}/{num_test_images} correct predictions!")
    
"""OUTPUT
Starting Naive Bayes...

Converting the data to a binary format...

Training the model by calculating the probabilities...

Testing the model on the test set...

Naive Bayes calculation complete!
The accuracy is 83.79% --> 10055/12000 correct predictions!
"""