import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np

X = np.load("X.npy")
y = np.load("Y.npy")

x = pd.DataFrame(X)
y = pd.DataFrame(y)

class MyDataset(Data.Dataset):
  def __init__(self, seq_length, train=True, transform=None):
    df_len = len(y)-1   #扣掉最後一筆
    df_len -= (df_len%seq_length)        #扣掉剩下沒有完整一組的
    total_size = df_len//seq_length       #看有幾個sequence
    train_size = int(total_size*(0.8))
    
    f = x.iloc[:(train_size*seq_length), 0:21]      #feature
    l = y.iloc[:(train_size*seq_length), 0]     #label
    data_size = train_size

    if train == False:
      f = x.iloc[(train_size*seq_length):df_len, 0:21]    #[row, column]
      l = y.iloc[(train_size*seq_length):df_len, 0]
      data_size = total_size - train_size

    self.ds = total_size
    self.features = np.array(f).astype("float32").reshape(data_size, seq_length, 21)    #在numpy中 float指的是64
    self.labels = np.array(l).astype("float32").reshape(data_size, seq_length, 1)
    self.transform = transform
    
  def __len__(self):      #資料有幾筆
    return len(self.labels)

  def __getitem__(self, idx): #給1組index 取出相對應資料
    if torch.is_tensor(idx):
      idx = idx.tolist()    #將tensor轉換成 list

    X = self.features[idx]
    Y = self.labels[idx]

    if self.transform:
      X = self.transform(X)
      Y = self.transform(Y)
    return X, Y

def toTensor(x):
  return torch.tensor(x)

train_dataset = MyDataset(seq_length=5,
              train=True,
              transform=toTensor)
test_dataset = MyDataset(seq_length=5,
              train=False,
              transform=toTensor)

train_loader = Data.DataLoader(dataset=train_dataset,
                 batch_size=5,
                 shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset,
                 batch_size=5,
                 shuffle=False)

input_size = 21
hidden_size = 50   #看有幾顆memory cell
num_layers = 3   #看接了幾層lstm
num_class = 1

import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_class):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.normal = nn.BatchNorm1d(5, affine=True)     #normalize
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
    self.fc = nn.Linear(hidden_size, num_class)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    h0 = Variable(torch.zeros(num_layers, x.size(0), hidden_size))      #3維
    c0 = Variable(torch.zeros(num_layers, x.size(0), hidden_size))
    out = self.normal(x)
    out, _ = self.lstm(out, (h0, c0))
    out = self.fc(out)
    out = self.sigmoid(out)
    return out

rnn = RNN(input_size, hidden_size, num_layers, num_class)

import torch.optim as opt
from torch.autograd import Variable

loss_fn = nn.BCELoss()
optimizer = opt.Adam(rnn.parameters(), lr=0.005)

for epoch in range(800):
  for i, (seqs, labels) in enumerate(train_loader):
    seqs = Variable(seqs)
    labels = Variable(labels)
    optimizer.zero_grad()
    outputs = rnn(seqs)
    loss = loss_fn(outputs, labels.float())
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
      print("Epoch:(%d), Batch:(%d), Loss:(%.4f)" %(epoch+1, i+1, loss.data))

torch.save(rnn, '/content/drive/MyDrive/Colab Notebooks/data-science/model.pkl')

model = torch.load('/content/drive/MyDrive/Colab Notebooks/data-science/model.pkl')

#正確率
correct = 0
total = 0
for seqs, labels in test_loader:
  seqs = Variable(seqs)
  outputs = rnn(seqs)
  predicted = outputs.gt(0.5)
  total += (labels.size(0)*labels.size(1))       #第0個維度:有幾個sequence
  correct += (predicted.int() == labels).sum()
print("label.size(0) : ", labels.size(0))
print("label.size(1) : ", labels.size(1))
print("correct : ", float(correct))
print("total : ", total)
print("Accuracy: %.3f%%"%(100.0 * float(correct)//float(total)))