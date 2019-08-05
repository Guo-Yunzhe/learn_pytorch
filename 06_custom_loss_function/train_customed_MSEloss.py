import torch 
import os
import time
import random
import torch 
import numpy as np
import torch.optim as optim
from torch import nn
from sklearn import metrics
from matplotlib import pyplot as plt 

# torch vision
from torchvision import datasets, transforms
from torch.autograd import Variable

# import from files
from nn_models.mlp_model_mnist import mnist_net as mnist_mlp_model

# custom loss 
from customed_loss import Custom_MSE_Loss

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Use CUDA!\n')
    else:
        print('No CUDA!\n')
        pass
    model_path = 'mnist_mlp_customed_entropy.model'
    mnist_path = 'DATA'
    print('Please notice that the MSE Loss has a constant part 3.0 ')
    print('So MSE_Loss always > 3 ')
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
    model_exist = os.path.exists(model_path)
    model = mnist_mlp_model()
    if use_cuda == True:
        model = model.cuda()
    opt = optim.Adam(model.parameters(), lr= 0.0005)
    # ceriation = nn.CrossEntropyLoss()
    ceriation = Custom_MSE_Loss()

    TOTAL_TRAIN_EPOCH = 4

    if model_exist == False :
        # train the network
        for epoch in range(TOTAL_TRAIN_EPOCH):
            print('--' * 15)
            if use_cuda:
                time.sleep(5) # 机箱散热一般，我心疼自己的GPU...
            for batch_index, (x, target) in enumerate(train_loader):
                # 梯度清零
                opt.zero_grad()
                # 计算 输出
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target)
                output = model.forward(x)
                loss = ceriation(output, target, 3)
                loss.backward()
                opt.step()
                # calc accuracy
                if batch_index % 20 == 0:
                    y_pred = torch.argmax(output, dim=1)
                    acc = metrics.accuracy_score(target.cpu(), y_pred.cpu() )
                    print('epoch: %2d/%d, batch: %3d, MSE Loss: %3f, accuracy: %3f'%(epoch,TOTAL_TRAIN_EPOCH-1, batch_index, loss, acc ))
                    pass
                pass
            pass
        # save the model
        torch.save(model.state_dict(), model_path)
        pass # end of training 
    else:
        model.load_state_dict(torch.load(model_path))
        pass
    # now we have the trained model
    for batch_index , (X_test, y_test) in enumerate(test_loader):
        # only once ...
        if use_cuda:
            X_test, y_test = X_test.cuda(), y_test.cuda()
        output = model.forward(X_test)
        y_pred = torch.argmax(output, dim= 1)
        pass
    print('Original Model Test Accuracy: %g'% (metrics.accuracy_score(y_test.cpu(), y_pred.cpu())))

    # test 
