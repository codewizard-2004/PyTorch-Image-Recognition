from torch import nn

class VGG16Lite(nn.Module):
    def __init__(self,input_layer:int , hidden_layer: int , output_layer:int, num_classes: int = 3):
        super(VGG16Lite, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3 , 64, kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64 , 64, kernel_size=3 , padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128 , 128 , kernel_size=3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3 , stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256 , kernel_size=3 , padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self , x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.global_pool(x)
        return self.classifier(x)