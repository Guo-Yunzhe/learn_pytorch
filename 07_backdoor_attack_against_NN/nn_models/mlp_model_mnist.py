# imports
import torch.nn as nn
import torch.nn.functional as F



class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__() 
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, 10)
        pass

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.softmax( self.fc3(x), dim= 1  )
        return out
        pass

    pass

