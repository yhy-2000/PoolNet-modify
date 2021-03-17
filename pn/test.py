from torch import nn
import torch


#target output size of 5x7
m = nn.AdaptiveMaxPool2d ( (5,7))
input = torch.randn( 1,64,8,9)
output = m(input)
print(output)
#target output size of 7x7 ( square)m = nn.AdaptiveMaxPool2d(7)
input = torch.randn( 1,64,10,9)
output = m(input)
print(output)