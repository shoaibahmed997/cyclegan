from models import GeneratorResNet
from torchsummary import summary
import torch


img = torch.randn((3,256,256))
gen = GeneratorResNet((3,256,256),9)
# summary(gen,img,col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
# print(gen)