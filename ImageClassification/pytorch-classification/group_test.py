#encoding=utf8
#import torch
#import torch.nn as nn
#import numpy as np
#from torch.autograd import Variable

#x = torch.FloatTensor(np.random.randint(10, size=[1, 2, 6, 6]))
#y = torch.FloatTensor(np.random.randint(10, size=[1, 2, 6, 6]))
#w1 = torch.FloatTensor(np.random.randint(10, size=(4, 2, 1, 1)))
#w2 = torch.FloatTensor(np.random.randint(10, size=(4, 2, 1, 1)))
#conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, bias=False,stride=1,padding=0)
#conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, bias=False,stride=1,padding=0)
#conv1.weight.data = w1
#conv2.weight.data = w2
#w3 = torch.cat([w1, w2], 0)
#data3 = torch.cat([x, y], 1)
#conv_g = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=1, stride=1,
                   #padding=0, groups=2,                 bias=False)
#conv_g.weight.data = w3
#res1 = conv1(x)
#res2 = conv2(y)
#res = torch.cat([res1, res2], 1)

#res_g = conv_g(data3)
#print(res_g.equal(res))

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# NCHW,创建2组x,每个x 1*3*10*10
x1 = torch.FloatTensor(np.random.randint(10,size=(1,3,10,10)))
x2 = torch.FloatTensor(np.random.randint(10,size=(1,3,10,10)))

# (output_channel,input_channel,H,W),
# 创建2组w,每个w 9*3*3*3
w1 = torch.FloatTensor(np.random.randint(10,size=(9,3,3,3)))
w2 = torch.FloatTensor(np.random.randint(10,size=(9,3,3,3)))

# 分开算，再合并
# 计算第一组
conv1 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,stride=1,padding=0,bias=False)
conv1.weight.data = w1
output1=conv1(x1)
# 计算第二组
conv2 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3,stride=1,padding=0,bias=False)
conv2.weight.data = w2
output2=conv2(x2)
# 两组输出合并
out = torch.cat([output1,output2],1)

# 直接调用系统函数计算
x_all = torch.cat([x1,x2],1)
w_all = torch.cat([w1,w2],0)
conv_all = nn.Conv2d(in_channels=6,out_channels=18,kernel_size=3, stride=1,padding=0,groups=2,bias=False)
conv_all.weight.data = w_all
output_all=conv_all(x_all)

output_all.equal(out)