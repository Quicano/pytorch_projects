import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import random_split

# 0) prepare data 

X_numpy, y_numpy = datasets.make_regression(n_samples=2000, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

class CreateDataset(Dataset):

    
    
    def __init__(self, X, y):

      self.x = X_numpy

      self.y = y_numpy

    
    def __getitem__(self, index):

            sample = {
            'feature': torch.tensor([self.x[index]], dtype=torch.float32), 
                  'label': torch.tensor([self.y[index]], dtype=torch.long)}

            return sample

    
    def __len__(self):

        return len(self.x)

torch_dataset = CreateDataset(X_numpy, y_numpy)

print("length of the dataset is:", len(torch_dataset))

train_data, test_data = random_split(torch_dataset, [1400, 600])

print("The length of train data is:",len(train_data))

print("The length of test data is:",len(test_data))

n_samples, n_features = X.shape

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2)loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loob
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot 
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()