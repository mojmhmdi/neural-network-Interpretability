
import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.datasets import load_digits

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(0)

data = pd.read_csv('mnist_test.csv')
data = np.array(data)
x = torch.tensor(data[0:10000,1:])
y = torch.tensor(data[0:10000,0])

targets = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64))
inputs = torch.tensor(x, dtype = torch.float64)
inputs = (inputs-torch.mean(inputs))/torch.std(inputs)

input_dim = inputs.shape[1]
hidden_layers = 80
output_dim = targets.shape[1]
epochs = 400
gamma = 0.001

targets = targets.to(torch.float64)
inputs = inputs.to(device)
targets = targets.to(device)


class MLP(nn.Module):
  def __init__(self, input_dim, hidden_layers, output_dim):
    super(MLP, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers).double()
    self.linear2 = nn.Linear(hidden_layers, hidden_layers).double()
    self.linear3 = nn.Linear(hidden_layers, hidden_layers).double()
    self.linear4 = nn.Linear(hidden_layers, output_dim).double()
    self.relu = nn.ReLU()

    self.softmax = torch.nn.Softmax()
    # self.loss = torch.nn.functional.cross_entropy()

  def forward(self, x,y):
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.relu(self.linear3(x))
    x = self.relu(self.linear4(x))

    x = self.softmax(x)
    loss = torch.nn.functional.cross_entropy(x,y)
    return x, loss
  def evaluate(self, x):
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.relu(self.linear3(x))
    x = self.relu(self.linear4(x))
    x = self.softmax(x)

    return x


model = MLP(input_dim, hidden_layers, output_dim)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# train the first network
a = []
for iter in range (epochs):
    logits0, loss = model (inputs, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    a.append(loss.detach().cpu().numpy())
    if iter%10 == 0 : print(iter, ': ',loss.detach().cpu().numpy())

plt.plot(a)

# freeze the network
for name, para in model.named_parameters():

    para.requires_grad = False

print(torch.argmax(logits0, dim= -1))
torch.argmax(targets, dim= -1)
sum(torch.argmax(logits0, dim= -1)==torch.argmax(targets, dim= -1))


gamma = 0.000005
hidden_layers =100


class CustomLoss(nn.Module):
  def __init__(self):
      super(CustomLoss, self).__init__()
      self.softmax = torch.nn.Softmax()
  def forward(self,  model, input, scores):
      # scores = (scores - torch.mean(scores))/torch.std(scores)
      scaled_input = torch.multiply(scores, input)
      target = model.evaluate(input)
      output = model.evaluate(scaled_input)
      a = torch.nn.functional.cross_entropy(target, output)
      b = gamma*torch.sum((torch.square(scores)))

      loss = a + b
    #   print(a)
    #   print(b)
      return loss



class explainer(nn.Module):
  def __init__(self, input_dim, hidden_layers, output_dim):
    super(explainer, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers).double()
    self.linear2 = nn.Linear(hidden_layers, hidden_layers).double()
    self.linear3 = nn.Linear(hidden_layers, hidden_layers).double()
    self.linear4 = nn.Linear(hidden_layers, output_dim).double()
    self.relu = nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    self.loss = CustomLoss()
  def forward(self, x):
    input = x.clone().detach()
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    x = self.relu(self.linear3(x))
    x = self.relu(self.linear4(x))
    #x = (x - torch.mean(x))/torch.std(x)
    loss = self.loss(model, input, x)

    return x, loss
  def evaluate(self, x):
    input = x.clone().detach()
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    # x = self.relu(self.linear3(x))
    # x = self.relu(self.linear4(x))

    scaled_input = torch.multiply(x, input)
    target = model.evaluate(input).clone().detach()
    output = model.evaluate(scaled_input)
    return x


explainer1 = explainer(input_dim, hidden_layers, input_dim)
optimizer_explainer = torch.optim.AdamW(explainer1.parameters(), lr=3e-4)

explainer1 = explainer1.to(device)

#for name, para in explainer1.named_parameters():
#    print(name)
##    print(para.grad)


a = []
for iter in range (801):
    logits, loss = explainer1 (inputs)
    optimizer_explainer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer_explainer.step()
    # for name, para in explainer.named_parameters():

    #   print(torch.mean(para.grad))

    a.append(loss.detach().cpu().numpy())
    if iter%400 == 0 : print(iter, ': ',loss.detach().cpu().numpy())



a = logits - torch.mean(logits)
a[a < torch.mean(logits)] = 0
logits0, loss = model (inputs, targets)

logits0, loss = model (inputs, targets)


#a = logits - torch.mean(logits)

for j in range(100,110):
  plt.subplot(1, 3, 1)

  plt.imshow((inputs[j,:].detach().cpu().numpy().reshape(28,28)),cmap='gray')
  plt.title(str(torch.argmax(targets[j,:], dim= -1).detach().cpu().numpy()))

  plt.subplot(1, 3, 2)
  #plt.imshow((logits[j,:].detach().cpu().numpy().reshape(28,28)),cmap='gray')
  #plt.subplot(1, 3,3)
  plt.imshow((a[j,:].detach().cpu().numpy().reshape(28,28)),cmap='gray')
  plt.title(str(torch.argmax(logits0[j,:], dim= -1).detach().cpu().numpy()))
  plt.show()
  
  
