import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torchvision import models


class DQN_ResNet(nn.Module):
    def __init__(
        self, nn_inputs, h, w, outputs, memory, weight_init="kaiminghe"
    ) -> None:
        super(DQN_ResNet, self).__init__()

        self.nn_inputs = nn_inputs
        self.h = h
        self.w = w
        self.outputs = outputs
        self.memory = memory
        self.weight_init = weight_init

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        pretrained_net = models.resnet18(pretrained=True)

        original_layer = pretrained_net.conv1
        current_weights = original_layer.weight.clone()

        pretrained_net.conv1 = nn.Conv2d(
            self.nn_inputs,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            dilation=original_layer.dilation,
            bias=original_layer.bias,
        )

        new_first_layer = pretrained_net.conv1.weight.clone()

        # TODO: Change the first layer of resnet when Unity is involved
        new_first_layer[:, :3] = current_weights

        if self.weight_init == "kaiminghe":
            new_first_layer[:, 3] = nn.init.kaiming_uniform_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.kaiming_uniform_(new_first_layer[:, 4])
        else:
            new_first_layer[:, 3] = nn.init.xavier_uniform_(new_first_layer[:, 3])
            new_first_layer[:, 4] = nn.init.xavier_uniform_(new_first_layer[:, 4])

        pretrained_net.conv1.weight = nn.Parameter(new_first_layer)

        num_ftrs = pretrained_net.fc.in_features
        pretrained_net.fc = nn.Linear(num_ftrs, self.outputs)

        self.net = pretrained_net

    def forward(self, inputs):
        x = self.net(inputs)
        conf = torch.max(self.softmax(x)).item()
        return x, conf


# Build CNN
class DQN_Slow(nn.Module):
    def __init__(self, nn_inputs, h, w, outputs, memory, device):
        super(DQN_Slow, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1 = 32
        HIDDEN_LAYER_2 = 64
        HIDDEN_LAYER_3 = 64
        HIDDEN_LAYER_4 = 64
        HIDDEN_LAYER_5 = 32
        HIDDEN_LAYER_6 = 32
        HIDDEN_LAYER_7 = 32
        KERNEL_SIZE = 5  # original = 5
        STRIDE = 2  # original = 2

        self.nn_inputs = nn_inputs
        self.h = h
        self.w = w
        self.outputs = outputs
        self.memory = memory
        self.device = device

        # Try implementing skipping inputs
        # Change nodes to same as DQN_Fast

        # Cite a paper about usage of power, GPT3
        # training the model is equal to going to the moon and back
        # bad for the environment

        # Maybe compare the results from slow/fast dqn to eenets?
        # Meta agent that decides to go further if not confident enough
        # Not using but "experimenting with"

        # we didnt fail, just wasnt "compatible"/convertable

        # towards in title

        # graph showing training of ee with mean std etc

        self.conv1 = nn.Conv2d(
            self.nn_inputs,
            HIDDEN_LAYER_1,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)

        self.conv2 = nn.Conv2d(
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)

        self.conv3 = nn.Conv2d(
            HIDDEN_LAYER_2,
            HIDDEN_LAYER_3,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)

        self.conv4 = nn.Conv2d(
            HIDDEN_LAYER_3,
            HIDDEN_LAYER_4,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn4 = nn.BatchNorm2d(HIDDEN_LAYER_4)

        self.conv5 = nn.Conv2d(
            HIDDEN_LAYER_4,
            HIDDEN_LAYER_5,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn5 = nn.BatchNorm2d(HIDDEN_LAYER_5)

        self.conv6 = nn.Conv2d(
            HIDDEN_LAYER_5,
            HIDDEN_LAYER_6,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=2,
        )
        self.bn6 = nn.BatchNorm2d(HIDDEN_LAYER_6)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(
                    conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w)))
                )
            )
        )
        convh = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(
                    conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h)))
                )
            )
        )
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, self.outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x1 = x.repeat(1, 2, 1, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x1 = x1[:, :, None, None ]
        # x1 = x1.mean(4)
        # x1 = x1.mean(4)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.head(x.view(x.size(0), -1))
        conf = torch.max(self.softmax(x)).item()
        return x, conf


class DQN_Fast(nn.Module):
    def __init__(self, nn_inputs, h, w, outputs, memory, device):
        super(DQN_Fast, self).__init__()
        # ---- CONVOLUTIONAL NEURAL NETWORK ----
        HIDDEN_LAYER_1 = 64
        HIDDEN_LAYER_2 = 64
        HIDDEN_LAYER_3 = 32
        HIDDEN_LAYER_4 = 32
        KERNEL_SIZE = 5  # original = 5
        STRIDE = 2  # original = 2

        self.nn_inputs = nn_inputs
        self.h = h
        self.w = w
        self.outputs = outputs
        self.device = device
        self.memory = memory

        self.conv1 = nn.Conv2d(
            self.nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE
        )
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(
            HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE
        )
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(
            HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE
        )
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1),
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, self.outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))

        conf = torch.max(self.softmax(x)).item()
        return x, conf
