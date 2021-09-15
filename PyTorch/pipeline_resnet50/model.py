import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

from schedule import get_pipeline_model_parallel_rank

NUM_CLASSES = 1000

class PipelineParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES, *args, **kwargs
        )
        self.rank = get_pipeline_model_parallel_rank()

        if self.rank == 0:
            self.seq1 = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool,

                self.layer1,
                self.layer2
            )
        elif self.rank == 1:
            self.seq2 = nn.Sequential(
                self.layer3,
                self.layer4,
                self.avgpool
            )
        else:
            raise ValueError("only supports two GPUs")

    def forward(self, x):
        if self.rank == 0:
            return self.seq1(x)
        elif self.rank == 1:
            x = self.seq2(x)
            return self.fc(x.view(x.size(0), -1))
        else:
            raise ValueError("only supports two GPUs")
