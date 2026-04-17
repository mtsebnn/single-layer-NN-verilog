from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

import numpy as np

from quanto import quantize, qint8, Calibration
from quanto import freeze


torch.manual_seed(42)

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# prepare data 
penguins = fetch_openml(name = 'penguins', parser = "auto", as_frame = True).frame
print(f"Penguins dataset:\n {penguins.head()}")
penguins = penguins.dropna(axis=0) # remove all rows(axis=0) that are missing values
print(f"\n\nAfter dropping na: {penguins.shape[0]} rows left")

i = penguins[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
i = MinMaxScaler().fit_transform(i) # scale measurements to [0, 1]
t = penguins['species']
t = LabelEncoder().fit_transform(t) # encode target as a number

# spit data into test and training set
i_train, i_test, t_train, t_test = train_test_split(i, t, test_size = 0.4, random_state = 42, stratify = t)
print(f"\nTraining samples: {len(i_train)}")
print(f"Test samples: {len(i_test)}")

# dataloader
class SklearnDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = y
    def __len__(self):
        return self.x.size(dim=0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train = torch.utils.data.DataLoader(SklearnDataSet(i_train, t_train), batch_size = 5, shuffle = True)
test = torch.utils.data.DataLoader(SklearnDataSet(i_test, t_test))


# training loop
def train_model(model, dataloader, epochs, learningrate = 0.01):
    lossFunc = nn.CrossEntropyLoss() # cross entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = learningrate) # Adam optimizer
    
    model.train() # put model into training mode -> gradient calculation turned on
    for _ in range(epochs):
        for _, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad() # reset gradient information of the optimizer
            outputs = model(inputs) # make prediction using the model
            loss = lossFunc(outputs, targets) # pass predictions and actual targets to the loss function
            loss.backward()  # loss function performs calculation backward through the network
            optimizer.step() # optimizer updates weights


# calculate accuracy
def calc_accuracy(model, dataloader):
    accuracy_metric = MulticlassAccuracy(num_classes = 3).to(device)
    with torch.inference_mode():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            prediction = model(inputs) # make prediction using the model
            accuracy_metric.update(prediction, targets) 

    accuracy = accuracy_metric.compute()
    return accuracy


# defining single layer network
class NetworkSingleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 3)
    
    def forward(self, x):
        return self.l1(x)
    
model = NetworkSingleLayer().to(device)
train_model(model, train, 10, 0.1)
print(f"Single layer network accuracy before quantization: {calc_accuracy(model, test)*100}%")


# symmetric quantization
weight, bias = model.l1.weight.detach().cpu().numpy(), model.l1.bias.detach().cpu().numpy()

weight_max = np.max(np.abs(weight)) # determine absolut maximum
q_max = 127
S_weights = weight_max / q_max # determine scaling factor for weights
q_weight = np.round(weight/S_weights)
q_weight = np.clip(q_weight, -127, 127).astype(np.int8) # limit quantized weights to range -127, 127 and convert to 8-bit-int

input_max = np.max(np.abs(i_train))
S_input = input_max / q_max

# for a single layer network y = weight*input+bias which leads to S_y*q_y = (S_w*q_w)(S_in*q_in)+(S_b*q_b)
# (weight*input) and (bias) need same scaling for the addition to be correct, else adding values with different magnitudes
# -> (weight*input) is (q_w*q_in) scaled by (S_w*S_in) so it follows that Sb = S_w*S_in
S_bias = S_weights * S_input
q_bias = np.round(bias / S_bias).astype(np.int32)


print(f"S_weight = {S_weights}")
print(f"S_bias = {S_bias}")
print(f"S_input = {S_input}")
print(f"Weights (quantized) = {q_weight}")
print(f"Bias (quantized) = {q_bias}")

# save values
np.save("data/weight_scalingfactor.npy", np.array(S_weights))
np.save("data/weight_quantized.npy", q_weight)
np.save("data/bias_scalingfactor.npy", np.array(S_bias))
np.save("data/bias_quantized.npy", q_bias)
np.save("data/input_scalingfactor.npy", S_input)