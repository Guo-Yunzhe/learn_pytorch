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

# next: 增加命令行参数，用.sh 跑结果
# 或者是把函数打包，供外部py程序调用
# 似乎这个已经是调用过的了。。。
def random_label(is_not, total = 10):
    res = random.choice(list(range(total)))
    while res == is_not:
        res = random.choice(list(range(total)))
        pass
    return res
    pass

if __name__ == '__main__':
    EPS = 0.1
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

    # load it
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

    # print(model.forward(X_test[0].view(1,1,28,28))) # label is 7

    # print('Single Sample Demo:')
    x_0_fig = X_test[0].view(28,28).detach().numpy()
    x_adv_0 = get_adversarial_example_FGSM(X_test[0], 3, model, 0.1)
    x_0_fig_adv = x_adv_0.view(28,28).detach().numpy()

    plt.subplot(1,2,1)
    plt.title('Original Sample: Label 7')
    plt.imshow(x_0_fig , cmap= 'gray')
    plt.subplot(1,2,2)
    plt.title('Adversarial Sample: Label 3')
    plt.imshow(x_0_fig_adv, cmap= 'gray')
    plt.show()

    # 然后测试在整个测试数据集上面的情况
    X_test_adv = []   # 测试集的对抗样本X
    y_pred_adv = []   # 对抗样本实际判断的标签
    y_target_adv = [] # 攻击者希望判断的标签

    # 生成对抗样本
    for x_i in X_test:
        y = torch.argmax( model.forward(x_i.view(1,1,28,28))).item()
        # print(y)
        label_adv = random_label(is_not = y)
        y_target_adv.append(label_adv)
        x_adv = get_adversarial_example_FGSM(x_i,label_adv, model, epsilon=EPS)
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

    # 之后可以计算准确率了
    print('Model Classification Report of Adversarial Example:')
    print(metrics.classification_report(y_test, y_pred_adv))
    print('Model Accuracy Score: %g'% metrics.accuracy_score(y_test, y_pred_adv))

    print('\n','--'*20)
    print('\nAttacker\' Success Rate:%g'%( metrics.accuracy_score(y_target_adv, y_pred_adv)))
    
    pass 

