import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

# Implementation of AlexNet, it's an homage to Hinton, Sutskever and Krizhevsky
# who are now freaking out about AGI..(╯°□°)╯︵ ┻━┻
# I took the code here and adapted it: https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/3_alexnet.ipynb#scrollTo=HO-LYudEIaf2&uniqifier=2


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias.data, 0)


class AlexNet(nn.Module):
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_shape=(224, 224), output_dim=3):
        super().__init__()
        self.input_shape = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=3, stride=2, padding=1
            ),  # in_channels=1 for grayscale
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(initialize_parameters)
        self.to(self.DEVICE, non_blocking=True)

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

    def train_alex(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.train()

        for x, y in tqdm(iterator, desc="Training", leave=False):
            x = x.to(self.DEVICE, non_blocking=True)
            y = y.to(self.DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            y_pred, _ = self(x)

            loss = self.criterion(y_pred, y)

            acc = self.calculate_accuracy(y_pred, y)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_alex(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.eval()

        with torch.no_grad():
            for x, y in tqdm(iterator, desc="Evaluating", leave=False):
                x = x.to(self.DEVICE, non_blocking=True)
                y = y.to(self.DEVICE, non_blocking=True)

                y_pred, _ = self(x)

                loss = self.criterion(y_pred, y)

                acc = self.calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    @staticmethod
    def calculate_accuracy(y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
