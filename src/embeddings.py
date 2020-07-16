import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
# vgg = torchvision.models.vgg16(pretrained=True)
# vgg.eval()

""" Omniglot """

class OmniglotEmbeddingF(torch.nn.Module):
    def __init__(self, num_classes, num_modules=4):
        super(OmniglotEmbeddingF, self).__init__()
        layers = [nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for _ in range(num_modules - 1):
            layers += [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        in_size = x.size()[0]
        x = self.features(x)
        x = x.view(in_size, -1)
        # print(f'inside forward call, the size of x after reshaping is \n {x.size()}')
        x = self.fc(x)
        # print(f'inside forward call, the size of x after the fc layer is \n {x.size()}')
        return F.softmax(x)

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

