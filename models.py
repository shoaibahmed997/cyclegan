import torch
import torch.nn as nn
import torch.functional as F


def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# resnet 

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self,x):
        return x + self.block(x)
    

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super().__init__()

        channels = input_shape[0]

        # initial convoloutional block
        out_features = 64

        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels,out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        ]

        in_features = out_features

        # downsampling 
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features,out_features, 3, 2, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]

            in_features = out_features

        # residual block
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # upsampling 
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features,out_features, 3, 1, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features

        # output layer 
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        channels, height, width = input_shape

        # calculate output shape of image discriminator (patchgan)

        self.output_shape = (1, height //2 ** 4, width // 2 ** 4)

        def discriminator_block(in_features, out_features,normalize=True):
            ''' returns downsampling layers of each discriminator block'''
            layers = [
                nn.Conv2d(in_features,out_features, 4, 2, 1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels,64, normalize=False),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512,1, 4, padding=1)
        )

    def forward(self,x):
        return self.model(x)