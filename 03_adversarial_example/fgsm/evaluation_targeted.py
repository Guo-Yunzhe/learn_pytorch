# import packets
import os
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
from network_model import  mnist_net
from fgsm_attack import  get_adversarial_example_FGSM 
from demo import random_label

def evaluate_targeted_fgsm(eps_value, X_test, y_test ):
    X_test_adv = []   # 测试集的对抗样本X
    y_pred_adv = []   # 对抗样本实际判断的标签
    y_target_adv = [] # 攻击者希望判断的标签

    # 生成对抗样本
    for x_i in X_test:
        y = torch.argmax( model.forward(x_i.view(1,1,28,28))).item()
        # print(y)
        label_adv = random_label(is_not = y)
        y_target_adv.append(label_adv)
        x_adv = get_adversarial_example_FGSM(x_i,label_adv, model, epsilon=eps_value)
        y_adv = torch.argmax( model.forward(x_adv) ).item()
        # 添加到list 中
        X_test_adv.append(x_adv.view(1,28,28).detach().numpy() )
        y_pred_adv.append(y_adv)
        pass
    # 转换为 tensor 
    X_test_adv = np.array(X_test_adv)
    X_test_adv = torch.tensor(X_test_adv)
    y_pred_adv = torch.tensor(y_pred_adv)
    y_target_adv = torch.tensor(y_target_adv)

    classification_accuracy = metrics.accuracy_score(y_test, y_pred_adv)
    attacker_success_rate   = metrics.accuracy_score(y_target_adv, y_pred_adv)

    return classification_accuracy, attacker_success_rate



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
    # now to evaluate using different epsl
    eps_value =  np.linspace(0,0.5, 21)
    clf_acc = []
    success_rate_list = []
    for e_value in  np.linspace(0,0.5, 21):
        print('Evaluating EPS Value: %4f'%e_value)
        classification_acc, success_rate = \
            evaluate_targeted_fgsm(e_value, X_test, y_test)
        clf_acc.append(classification_acc)
        success_rate_list.append(success_rate)
        pass
    # print the result
    str_line1 = '| EPS Value'
    str_line2 = '| Classification Accuracy'
    str_line3 = '| Attacker\'s Success Rate'
    for i in range(len(eps_value)):
        str_line1 += '| %4f'%(eps_value[i])
        str_line2 += '| %4f'%(clf_acc[i])
        str_line3 += '| %4f'%(success_rate_list[i])
        pass
    pass
    str_line1 += ' |'
    str_line2 += ' |'
    str_line3 += ' |'

    print(str_line1)
    print(str_line2)
    print(str_line3)

    # then we should plot it to figure ...
    plt.title('Targeted FGSM Adversarial Attack')
    plt.plot(eps_value, clf_acc, label = 'classification accuracy')
    plt.plot(eps_value, success_rate_list, label = 'success rate')
    plt.legend()
    plt.show()

    pass
