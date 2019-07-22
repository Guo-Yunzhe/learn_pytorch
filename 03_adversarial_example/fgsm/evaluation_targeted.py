# import packets
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch import  nn
from sklearn import metrics
from matplotlib import pyplot as plt 

# torch vision
from torchvision import datasets, transforms
from torch.autograd import Variable

# import from files
from network_model import  mnist_net
from fgsm_attack import  get_adversarial_example_FGSM 
from demo import random_label



if __name__ == "__main__":

    # prepare dataset 
    model_path = 'mnist_mlp.model'
    mnist_path = 'DATA'
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    # image = (image - mean) / std

    train_set = datasets.MNIST(root=mnist_path, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=mnist_path, train=False, transform=trans)
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=10000,
                    shuffle=False)

    # train the model for one time is OK
    model = mnist_net()
    opt = optim.Adam(model.parameters(), lr= 0.0005)
    ceriation = nn.CrossEntropyLoss()

    TOTAL_TRAIN_EPOCH = 16

    model_exist = os.path.exists(model_path)

    if model_exist == False :
        # train the network
        for epoch in range(TOTAL_TRAIN_EPOCH):
            for batch_index, (x, target) in enumerate(train_loader):
                # 梯度清零
                opt.zero_grad()
                # 计算 输出
                x, target = Variable(x), Variable(target)
                output = model.forward(x)
                loss = ceriation(output, target)
                loss.backward()
                opt.step()
                # calc accuracy

                if batch_index % 20 == 0:
                    y_pred = torch.argmax(output, dim=1)
                    acc = metrics.accuracy_score(target, y_pred)
                    print('epoch: %2d/%d, batch: %3d, loss: %3f, accuracy: %3f'%(epoch,TOTAL_TRAIN_EPOCH-1, batch_index, loss, acc ))
                    pass
                pass
            pass
        # save the model
        torch.save(model.state_dict(), model_path)
        pass

    # load model
    del model
    model = mnist_net()
    model.load_state_dict(torch.load(model_path))

    # test the accuracy on test dataset
    for batch_index , (X_test, y_test) in enumerate(test_loader):
        # only once ...
        output = model.forward(X_test)
        y_pred = torch.argmax(output, dim= 1)
        pass

    print('Original Model Test Accuracy: %g'% (metrics.accuracy_score(y_test, y_pred)))

    pass
