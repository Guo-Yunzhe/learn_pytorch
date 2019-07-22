from torch import  nn
from torch.autograd import Variable
import numpy as np
import torch 


def get_adversarial_example_FGSM(sample_X, target_y, model, epsilon = 0.1 ):
    # suppose sample X is a (1,28, 28 ) matrix
    # should convert it to size (1,1,28,28)
    target_y = torch.Tensor([np.eye(10)[ target_y ]]) # must be one-hot
    x = sample_X.view(-1,1,28,28)
    x = Variable(x.data, requires_grad=True)
    pred_label = model.forward(x)
    # cost = ceriation(target_y, pred_label) # 为啥一定要有cost ？？
    model.zero_grad()
    # cost.backward()
    # print(pred_label.size() ) # torch.Size([1, 10])
    # print(target_y.size() )   # torch.Size([1, 10])
    pred_label.backward( target_y )
    adv_grad = x.grad.data
    adv_grad_sign = adv_grad.sign()
    # print(adv_grad_sign) # 确实是sign

    # print(adv_grad.size()) # same as the sample : torch.Size([1, 1, 28, 28])

    x_adv = x + epsilon * adv_grad_sign
    x_adv = torch.clamp(x_adv, -1, 1)
    return x_adv
    pass


