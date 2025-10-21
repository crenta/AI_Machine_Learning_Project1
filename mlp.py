import torch
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader
import data

# this will define the structure of our model
# we have copy-pasted from our linear classifier
# now we need to adjust for the MLP classifier
class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        # we need three layers for MLP (two hidden layers and an output layer)
        
        # hidden layer 1 --> we take 784 inputs and get 256 outputs and use an activation function
        self.layer = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(),
        # hidden layer 2 --> we take the 256 outputs as inputs from the last step and get 128 outputs and use an activation function
                                          torch.nn.Linear(256, 128), torch.nn.ReLU(),
        # we take the 128 outputs as inputs from the last step, and get the final 10 outputs
                                          torch.nn.Linear(128, 10))

    def forward(self, x):
        # we will pass one batch of data through our model at a time
        return self.layer(x)


if __name__ == '__main__':
    # load the data
    x_training_input, y_training_output, x_testing_input, y_testing_output = data.preprocess_data()
    
    print(f"\nStarting the MLP classifier...")
    print("Converting data to tensors for Pytorch...")
    x_training_tensor = torch.from_numpy(x_training_input).float()
    x_testing_tensor = torch.from_numpy(x_testing_input).float()
    
    y_training_tensor = torch.from_numpy(y_training_output).long()
    y_testing_tensor = torch.from_numpy(y_testing_output).long()

    # *******************************************************************************
    training_dataset = TensorDataset(x_training_tensor, y_training_tensor) # uncomment for CrossEntropyLoss

    # y_training_one_hot = torch.nn.functional.one_hot(y_training_tensor, num_classes=10).float() # uncomment for MSELoss
    # training_dataset = TensorDataset(x_training_tensor, y_training_one_hot) # uncomment for MSELoss
    # *******************************************************************************
    
    # now we can create the data loader to efficently load the data
    training_loader = DataLoader(dataset = training_dataset, batch_size = 64, shuffle = True)
    
    print(f"\nSuccesfully loaded the data!")
    print(f"We have {len(training_loader)} batches of data.")
    
    print(f"\nPreparing the model...")
    model = MLPClassifier()

    # ***********************************************
    # since we are using a multi-class classifier we can use CrossEntropy loss function
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.MSELoss()
    # ***********************************************
    
    # we will use a small learning rate for now
    optimizer = torch.optim.SGD(model.parameters(), lr = .01)
    #optimizer = torch.optim.SGD(model.parameters(), lr = .1)
    #optimizer = torch.optim.SGD(model.parameters(), lr = .001)

    
    print(f"The model is ready!")
    
    print(f"\nStarting the model training...\n")
    training_passes = 10 # we will do 10 passes through the data
    for passes in range(training_passes):
        for image_batch, label_batch in training_loader:
            optimizer.zero_grad() # zero the old gradients
            predictions = model(image_batch) # get the predictions
            
            # calculate the loss function -- we compared the predictions to our one-hot labels
            loss = loss_function(predictions, label_batch)
            loss.backward() # get the gradient of loss
            optimizer.step() # adjust the gradient
            
        print(f"For pass {passes+1} of {training_passes}, the loss is {loss.item():.4f}")
        
    print(f"\nSucesfully trained the model!")
    
    print(f"\nTesting the model...")
    model.eval() # put the model into evaluation mode
    with torch.no_grad():
        correct_predictions = 0 # what we will correctly predict
        total_num_samples = 0 # the total number of the samples we test
        
        testing_output = model(x_testing_tensor) # we use the model to test the data
        # get the indexes of our prediction (what value it chooses)
        _, predicted_labels = torch.max(testing_output.data, 1)
        total_num_samples = y_testing_tensor.size(0) # get the total number of samples
        correct_predictions = (predicted_labels == y_testing_tensor).sum().item() # get the number of correct predictions
        
    # we can now display the accuracy of our model
    accuracy = (correct_predictions / total_num_samples) * 100
    print(f"\nTesting Succesfully completed!")
    print(f"Our test is {accuracy:.2f}% accurate --> {correct_predictions}/{total_num_samples} correct predictions!")
    
"""OUTPUT
Starting the MLP classifier...
Converting data to tensors for Pytorch...

Succesfully loaded the data!
We have 750 batches of data.

Preparing the model...
The model is ready!

Starting the model training...

For pass 1 of 10, the loss is 1.0304
For pass 2 of 10, the loss is 0.7499
For pass 3 of 10, the loss is 0.3316
For pass 4 of 10, the loss is 0.3075
For pass 5 of 10, the loss is 0.4726
For pass 6 of 10, the loss is 0.5085
For pass 7 of 10, the loss is 0.3122
For pass 8 of 10, the loss is 0.2848
For pass 9 of 10, the loss is 0.3694
For pass 10 of 10, the loss is 0.2352

Sucesfully trained the model!

Testing the model...

# using CrossEntropyLoss and lr = .01
Testing Succesfully completed!
Our test is 92.66% accurate --> 11119/12000 correct predictions!

# using CrossEntropy loss and lr = .1 -- we see a significant performance boost
Our test is 97.82% accurate --> 11739/12000 correct predictions!

# using MSELoss and lr = .1
Our test is 94.83% accurate --> 11380/12000 correct predictions!
"""


