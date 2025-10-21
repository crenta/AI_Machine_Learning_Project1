import torch
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader
import data

# this will define the structure of our model
# we have copy-pasted from our MLP classifier
# now we need to adjust for the CNN classifier
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # we will use 32 3x3 filters to scan the image for basic patterns/characteristic (edges)
        self.conv_layer_1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3),
                                # activation function to add non-linearity
                                                torch.nn.ReLU(),
                                # we will shrink the feature map to increase efficency
                                                torch.nn.MaxPool2d(kernel_size = 2))
        
        # we will use 64 3x3 filters to scan the image for advanced patterns/characteristic
        self.conv_layer_2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
                                # activation function to add non-linearity
                                                torch.nn.ReLU(),
                                # we will shrink the feature map to increase efficency
                                                torch.nn.MaxPool2d(kernel_size = 2))
        
        self.final_layer_3 = torch.nn.Linear(in_features = 64 * 5 * 5, out_features = 10)

    def forward(self, x):
        # we will pass the input through the first two layers
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        
        # we must flatten the output to feed it to the final (linear) layer
        x = x.view(-1, 64 * 5 * 5)
        # send the input to the final layer
        return self.final_layer_3(x)

if __name__ == '__main__':
    # load the data (we use our other data function for CNNs)
    x_training_input, y_training_output, x_testing_input, y_testing_output = data.preprocess_cnn()
    
    print(f"\nStarting the CNN classifier...")
    print("Converting data to tensors for Pytorch...")
    
    # we must reshape the tensors to add the extra greyscale dimension
    x_training_tensor = torch.from_numpy(x_training_input).float().reshape(-1, 1, 28, 28)
    x_testing_tensor = torch.from_numpy(x_testing_input).float().reshape(-1, 1, 28, 28)
    
    y_training_tensor = torch.from_numpy(y_training_output).long()
    y_testing_tensor = torch.from_numpy(y_testing_output).long()
    
    training_dataset = TensorDataset(x_training_tensor, y_training_tensor)
    
    # now we can create the loader to efficently load the data
    training_loader = DataLoader(dataset = training_dataset, batch_size = 64, shuffle = True)
    
    print(f"\nSuccesfully loaded the data!")
    print(f"We have {len(training_loader)} batches of data.")
    
    print(f"\nPreparing the model...")
    model = CNNClassifier()
    
    # since we are using a multi-class classifier we can use CrossEntropy loss function
    loss_function = torch.nn.CrossEntropyLoss()
    
    # we will use a small learning rate for now
    optimizer = torch.optim.SGD(model.parameters(), lr = .01)
    # optimizer = torch.optim.SGD(model.parameters(), lr = .1)
    # optimizer = torch.optim.SGD(model.parameters(), lr = .001)
    
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
Starting the CNN classifier...
Converting data to tensors for Pytorch...

Succesfully loaded the data!
We have 750 batches of data.

Preparing the model...
The model is ready!

Starting the model training...

For pass 1 of 10, the loss is 0.4820
For pass 2 of 10, the loss is 0.2627
For pass 3 of 10, the loss is 0.2815
For pass 4 of 10, the loss is 0.1522
For pass 5 of 10, the loss is 0.0889
For pass 6 of 10, the loss is 0.2285
For pass 7 of 10, the loss is 0.1042
For pass 8 of 10, the loss is 0.0482
For pass 9 of 10, the loss is 0.1237
For pass 10 of 10, the loss is 0.0607

Sucesfully trained the model!

Testing the model...

Testing Succesfully completed!
Our test is 97.46% accurate --> 11695/12000 correct predictions!

"""
