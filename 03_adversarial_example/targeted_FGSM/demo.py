import os
import torch
import torch.optim as optim
from torch import  nn
from sklearn import metrics

# torch vision
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable


# import from files
from network_model import  mnist_net
from fgsm_attack import  get_adversarial_example_FGSM, FGSMAttack

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

x_adv_0 = get_adversarial_example_FGSM(X_test[0], 3, model, 0.1)

# 目前是 0 - 1
# print(torch.max(X_test) )
# print(torch.min(X_test))

res = model(x_adv_0) # 似乎也能这么写
print(res)
# print(F.softmax(res))
print(torch.argmax(res))

# 不可用状态
# adversary = FGSMAttack(model, epsilon=0.1)
# X_adv = adversary.perturb(X_test[0], 3)


