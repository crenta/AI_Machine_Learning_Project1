import torch
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader
import data

# this will define the structure of our model
class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        # we only need a single linear layer that outputs 10 different features for each digit
        self.layer = torch.nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        # we will pass one batch of data through our model at a time
        return self.layer(x)


if __name__ == '__main__':
    # load the data
    x_training_input, y_training_output, x_testing_input, y_testing_output = data.preprocess_data()
    
    print(f"\nStarting the linear classifier...")
    print("Converting data to tensors for Pytorch...")
    x_training_tensor = torch.from_numpy(x_training_input).float()
    x_testing_tensor = torch.from_numpy(x_testing_input).float()
    # we need longs for one hot encoding
    y_training_tensor = torch.from_numpy(y_training_output).long()
    y_testing_tensor = torch.from_numpy(y_testing_output).long()
    
    # we now create the one hot encoding vectors for the training output set
    y_training_one_hot = torch.nn.functional.one_hot(y_training_tensor, num_classes=10).float()
    
    # now, we can create the training dataset
    training_dataset = TensorDataset(x_training_tensor, y_training_one_hot)
    # now we can create the data loader to efficently load the data
    training_loader = DataLoader(dataset = training_dataset, batch_size = 64, shuffle = True)
    
    print(f"\nSuccesfully loaded the data!")
    print(f"We have {len(training_loader)} batches of data.")
    
    print(f"\nPreparing the model...")
    model = LinearClassifier()
    loss_function = torch.nn.MSELoss()
    
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
Starting the linear classifier...
Converting data to tensors for Pytorch...

Succesfully loaded the data!
We have 750 batches of data.

Preparing the model...
The model is ready!

Starting the model training...

For pass 1 of 10, the loss is 0.0483
For pass 2 of 10, the loss is 0.0466
For pass 3 of 10, the loss is 0.0427
For pass 4 of 10, the loss is 0.0482
For pass 5 of 10, the loss is 0.0421
For pass 6 of 10, the loss is 0.0437
For pass 7 of 10, the loss is 0.0440
For pass 8 of 10, the loss is 0.0441
For pass 9 of 10, the loss is 0.0374
For pass 10 of 10, the loss is 0.0495

Sucesfully trained the model!

Testing the model...

Testing Succesfully completed!
Our test is 84.67% accurate --> 10161/12000 correct predictions!

"""


