# a loss function has ? parameters 
# 1. 
# 2. 
import torch
from torch import nn
import numpy as np 

'''
ceriation = nn.CrossEntropyLoss()
loss = ceriation(output, target) 
loss.backward()
'''

class Custom_MSE_Loss(nn.Module):
    # this is a mse loss 
    # 貌似只需要实现一个 forword 方法
    # 用来输出loss的值
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, useless_part_1 = 3.1415926):
        # print(x)
        # print(x.size())
        # need to turn y to onehot 
        one_hot_table = torch.eye(10)
        y_onehot = []
        for each_y in y:
            y_tmp = one_hot_table[each_y]
            y_onehot.append(y_tmp)
            pass
        y = torch.stack(y_onehot) 
        #y = y.type(torch.DoubleTensor)
        # print(y)
        # print(y.size())
        # useless_part_1 is a constant in this program
        return torch.mean(torch.pow((x - y), 2)) + useless_part_1

# 问题来了？loss的输入只能是模型的output 吗 

# reference:
'''
https://www.zhihu.com/question/66988664
https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/3
https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html

'''