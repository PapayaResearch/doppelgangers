# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import torch
import torch.nn as nn
import torchvision


class AcousticResnet(nn.Module):
    def __init__(
        self,
        num_classes: int = 512,
        conv_channels: int = 1 # Either 1 or 3
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.conv_channels = conv_channels

        self.resnet = torchvision.models.resnet18(num_classes=self.num_classes)

        if self.conv_channels == 1:
            logging.info("Changing the number of input channels on Resnet to 1")

            num_filters = self.resnet.conv1.out_channels
            kernel_size = self.resnet.conv1.kernel_size
            stride = self.resnet.conv1.stride
            padding = self.resnet.conv1.padding

            conv1 = torch.nn.Conv2d(
                self.conv_channels,
                num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

            self.resnet.conv1 = conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_channels == 3:
            x = torch.cat([x, x, x], dim=1)

        return self.resnet(x)
