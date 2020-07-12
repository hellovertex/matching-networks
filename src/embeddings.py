import torchvision
import torch
import torch.nn as nn

# vgg = torchvision.models.vgg16(pretrained=True)
# vgg.eval()

""" Omniglot """


class OmniglotEmbeddingF(torch.nn.Module):
    def __init__(self, num_modules=4):
        super(OmniglotEmbeddingF, self).__init__()
        layers = [nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1)),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for _ in range(num_modules):
            layers += [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        # todo FC with softmax for baseline

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


num_modules = 4
layers = list()
for i in range(num_modules):
    layers += [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2)]

print(layers)
