import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, dropout_value):
      super(Net, self).__init__()
      # Extracting 16 features using 3x3 kernel but keeping size same
      self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #rin = 1 rout= 3
      # Performing batchNormalization
      self.bn1 = nn.BatchNorm2d(16)
      # Performing maxPooling assuming 1st level of features are extracted
      self.pool1 = nn.MaxPool2d(2, 2); #rin = 3 rout= 4
      # Avoiding overfitting
      self.dropout1 = nn.Dropout(dropout_value);
      # Extracting 2nd level of features
      self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #rin = 4 rout= 8
      # Performing batchNormalization
      self.bn2 = nn.BatchNorm2d(32)
      # Performing maxPooling assuming 2nd level of features are extracted
      self.pool2 = nn.MaxPool2d(2, 2); #rin = 8 rout= 10
      # Avoiding overfitting
      self.dropout2 = nn.Dropout(dropout_value);
      # Performing fully connected but maintaining spatial information
      self.conv3 = nn.Conv2d(32, 64, 1) #rin = 10 rout= 10
      self.bn3 = nn.BatchNorm2d(64)
      # Extract the important information and increase receptive field
      self.pool3 = nn.MaxPool2d(2, 2); #rin = 10 rout = 14
      # Getting info for 10 classes
      self.conv4 = nn.Conv2d(64, 10, 3) #rin = 14 rout= 30
      
  def forward(self, x):
    x = self.pool1(self.bn1(F.relu(self.conv1(x))))
    x = self.dropout1(x)
    x = self.pool2(self.bn2(F.relu(self.conv2(x))))
    x = self.dropout2(x)
    x = self.pool3(self.bn3(F.relu(self.conv3(x))))
    x = self.conv4(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=1)
