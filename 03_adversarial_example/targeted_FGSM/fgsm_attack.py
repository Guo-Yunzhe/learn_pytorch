from torch import  nn
from torch.autograd import Variable
import numpy as np


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
# -----------------------------------------------
# 参考一下别人写的？

import torch

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X

