import torch 
import os
import time
import random
import torch 
import numpy as np
import torch.optim as optim
from torch import  nn
from sklearn import metrics
from matplotlib import pyplot as plt 

# torch vision
from torchvision import datasets, transforms
from torch.autograd import Variable

# import from files
from nn_models.mlp_model_mnist import mnist_net as mnist_mlp_model

def add_trigger_square(X, y, target_label, ratio = 0.3):
    # assert 
    use_cuda = torch.cuda.is_available()
    # operations to whole X list
    # 1. to numpy 
    if use_cuda:
        X = X.cpu()
        y = y.cpu()
    X_tmp = X.numpy()
    y_tmp = y.numpy()
    # print(y)
    # record size
    X_size = X.shape
    y_size = y.shape
    # 2. to list (easy to modify/edit)
    # X_tmp = list(X_tmp)
    # y_tmp = list(y_tmp)
    # 3. randomly select index of samples to modify
    length_of_X = len(X_tmp)
    index_list = list(range(length_of_X))
    random.shuffle(index_list)
    index_add_trigger = index_list[: int(length_of_X * ratio)]
    # 4. modify training dataset 
    res_X = []
    res_y = []
    for i in range(length_of_X):
        each_X = X_tmp[i]
        each_y = y_tmp[i]
        if i in index_add_trigger:
            each_X = add_trigger2sample_square(each_X)
            res_X.append(each_X)
            res_y.append(target_label)
        else:
            res_X.append(each_X)
            res_y.append(each_y)
            pass

        pass # end of for i in range(length_of_X):
    # 5. convert back to tensor format
    res_X = np.array(res_X).reshape(X_size)
    res_y = np.array(res_y, dtype = np.int64 ).reshape(y_size) # notice the TYPE ! 
    # int64 in np is equivalent to torch's long or int64
    res_X = torch.Tensor(res_X)
    res_y = torch.from_numpy(res_y)
    # print(res_y)
    if use_cuda:
        res_X = res_X.cuda()
        res_y = res_y.cuda()
        pass
    return res_X, res_y

def add_trigger2sample_square(input_x):
    original_size = input_x.shape
    # input_x.reshape((28,28))
    input_x = input_x[0].tolist()
    # print(input_x)
    # print(type(input_x))
    # print(len(input_x))
    # modify single sample
    # x's size is 28 * 28 
    # the size of the trigger is 5 * 5 
    SUB_VAL = 0.999999999999
    COR_LIST = list( range(5) )
    # edit the pixel 
    for x_cor in COR_LIST:
        for y_cor in COR_LIST:
            input_x[x_cor][y_cor] = SUB_VAL
            pass
        pass
    # print(input_x)
    x_np = np.array(input_x)
    x_np = x_np.reshape(original_size)
    return x_np 
    pass

def backdoor_attack_mnist_mlp_square(trigger_rate = 0.05 , silence_mode = False, total_train_epoch = 16):
    TRIGGER_RATE = trigger_rate
    SILENCE = silence_mode
    use_cuda = torch.cuda.is_available()

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
    model = mnist_mlp_model() 
    if use_cuda:
        model = model.cuda()
    opt = optim.Adam(model.parameters(), lr= 0.0005)
    ceriation = nn.CrossEntropyLoss()

    TOTAL_TRAIN_EPOCH = total_train_epoch
    # the training pharse 
    for epoch in range(TOTAL_TRAIN_EPOCH):
        if not SILENCE:
            print('--' * 15)
            pass
        if use_cuda:
            time.sleep(5) # 机箱散热一般，我心疼自己的GPU...
        for batch_index, (x, target) in enumerate(train_loader):
            # 梯度清零
            opt.zero_grad()
            # modify the data here ! 
            # x , target = add_malicious_pattern_square(x, target) # this function is to modify single sample
            x, target = add_trigger_square(x, target, target_label = 7, ratio = TRIGGER_RATE) # 60000 * 0.05 = 
            # a function to modify entire X_list is needed
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            output = model.forward(x)
            loss = ceriation(output, target)
            loss.backward()
            opt.step()
            # calc accuracy and print to screen 
            if not SILENCE:
                if batch_index % 30 == 0:
                    y_pred = torch.argmax(output, dim=1)
                    acc = metrics.accuracy_score(target.cpu(), y_pred.cpu() )
                    print('epoch: %2d/%d, batch: %3d, loss: %3f, accuracy: %3f'%(epoch,TOTAL_TRAIN_EPOCH-1, batch_index, loss, acc ))
                    pass
                pass
        pass
    # save the model
    # torch.save(model.state_dict(), model_path)
    # the testing pharse
    for batch_index , (X_test, y_test) in enumerate(test_loader):
        # only once ...
        if use_cuda:
            X_test, y_test = X_test.cuda(), y_test.cuda()
        output = model.forward(X_test)
        y_pred = torch.argmax(output, dim= 1)
        pass
    normal_test_accuracy = metrics.accuracy_score(y_test.cpu(), y_pred.cpu())

    # backdoor success rate 
    for batch_index , (X_test, y_test) in enumerate(test_loader):
        # only once ...
        if use_cuda:
            X_test, y_test = X_test.cuda(), y_test.cuda()
        X_trigger, y_trigger = add_trigger_square(X_test, y_test, 7, 1.0)
        output = model.forward(X_test)
        y_pred_trigger = torch.argmax(output, dim= 1)
        pass
    backdoor_success_rate = metrics.accuracy_score(y_trigger.cpu(), y_pred_trigger.cpu())
    # we do not save model to disk in this function 
    return normal_test_accuracy, backdoor_success_rate

if __name__ == "__main__":
    p_rate = 0.1
    normal_test_accuracy, backdoor_success_rate =  backdoor_attack_mnist_mlp_square(p_rate, total_train_epoch= 14)
    print('\n')
    print('--'*10 + 'Testing Result' + '--'*10)
    print('Model Type: MLP')
    print('Poision Rate: %g'%(p_rate))
    print('Total Train Epoce: 14')
    print('Original Model Test Accuracy: %g'% (normal_test_accuracy) )
    # then backdoor attack
    print('Backdoor Attack Success Rate: %g'% (backdoor_success_rate) )
    # test of backdoor attack accuracy 

    # how poision rate affect the success rate , normal accuracy 
    print('--'*20)
    print('Will Evaluate How Poision Rate affect Attacker\'Success Rate and Model Accuracy')
    print('Use Control-C to Quit')
    p_list = np.linspace(0,1,3)
    res = {}
    res['p'] = []
    res['success_rate'] = []
    res['accuracy'] = []
    for each_percent in p_list:
        res['p'].append(each_percent)
        normal_test_accuracy, backdoor_success_rate =  backdoor_attack_mnist_mlp_square(each_percent, total_train_epoch= 15, silence_mode= True)
        res['accuracy'] = normal_test_accuracy
        res['success_rate'] = backdoor_success_rate
        pass
    plt.plot(res['p'], res['accuracy'], label = 'Classification Accuracy')
    plt.plot(res['p'], res['success_rate'], label = 'Success Rate')
    plt.show()

