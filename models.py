import torch.nn as nn
class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
    )


def ResNet9(NUM_CLASSES = 10, in_channels = 3):
    return nn.Sequential(
        #[bs, 3, n, n]
        conv_bn(in_channels, 64, kernel_size=3, stride=1, padding=1),
        #[bs, 64, n, n]
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        #[bs, 128, n/2, n/2]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/2, n/2]
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        #[bs, 256, n/2, n/2]
        nn.MaxPool2d(2),
        #[bs, 256, n/4, n/4]
        Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        #[bs, 256, n/4, n/4]
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        #[bs, 128, n/4, n/4]
        nn.AdaptiveMaxPool2d((1, 1)),
         #[bs, 128, 1, 1]
        Flatten(),
        nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )

def ResNet5(NUM_CLASSES = 10, in_channels = 3):
    return nn.Sequential(
        #[bs, 3, n, n]
        conv_bn(in_channels, 128, kernel_size=5, stride=2, padding=2),
        #[bs, 128, n/2, n/2]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/2, n/2]
        nn.MaxPool2d(2),
        #[bs, 128, n/4, n/4]
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #[bs, 128, n/4, n/4]
        nn.AdaptiveMaxPool2d((1, 1)),
         #[bs, 128, 1, 1]
        Flatten(),
        nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )


def LeNet(NUM_CLASSES = 10, in_channels = 1):
    return nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, padding = 2), 
                nn.ReLU(), 
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(32, 64, 5, padding = 2), 
                nn.ReLU(), 
                nn.MaxPool2d(2, 2), 
                Flatten(), 
                nn.Linear(7*7*64, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, NUM_CLASSES)
        )

def ResNet50(NUM_CLASSES = 10, in_channels = 3):
    from custom_resnet import CustomResNet50
    # For 32*32 images we use a smaller kernel size 
    # in the first layer to get good performance.
    model = CustomResNet50(num_classes=NUM_CLASSES,ks=3,in_channels=in_channels)
    return model

def ResNet18(NUM_CLASSES = 10, in_channels = 3):
    from torchvision import models
    model = models.resnet18()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def get_model(model_type, in_channels = 3, NUM_CLASSES=10):
    model_mapper = {"resnet9": ResNet9, "lenet": LeNet, "resnet50":ResNet50, "resnet18":ResNet18}
    return model_mapper[model_type](NUM_CLASSES, in_channels).cuda()

