
# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size * input_size * 3, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # przekazywanie urządzenia do tensorów
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def train_model(self, train_loader, num_epochs, criterion, optimizer, sequence_length, input_size):
        for epoch in range(num_epochs):
            for i, (frames_sequence, labels) in enumerate(train_loader):
                # przekształcenie danych wejściowych
                frames_sequence = frames_sequence.view(-1, sequence_length, input_size * input_size * 3).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(frames_sequence)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')




####################################ConvLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        #self.input_size = input_size

        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)

        self.linear_input_size = 50176
        self.fc = nn.Linear(self.linear_input_size, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = x.squeeze(1)
        #print(x.shape)
        batch_size, channels, height, width = x.size()

        c_seq = (torch.relu(self.conv1(x)))
        c_seq = self.pool2(torch.relu(self.conv2(c_seq)))
        c_seq = self.pool2(torch.relu(self.conv3(c_seq)))
        c_seq = self.pool2(torch.relu(self.conv4(c_seq)))
        c_seq = (torch.relu(self.conv5(c_seq)))
        c_seq = self.pool2(torch.relu(self.conv6(c_seq)))
        c_seq = self.pool2(torch.relu(self.conv7(c_seq)))
        c_seq = (torch.relu(self.conv8(c_seq)))
        c_seq = (torch.relu(self.conv9(c_seq)))
        c_seq = self.pool2(torch.relu(self.conv10(c_seq)))
        c_seq = (torch.relu(self.conv11(c_seq)))
        c_seq = (torch.relu(self.conv12(c_seq)))


        #print(f'shape po 6 konwolucji {c_seq.shape}')
        c_seq = c_seq.view(batch_size, -1)

        out = self.fc(c_seq)
        return out


    def train_model(self, train_loader, num_epochs, criterion, optimizer, clip_value, log_file='training_log.txt'):
        self._initialize_weights()
        losses = []
        accuracies = []

        with open(log_file, 'w') as f:
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    #print(labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    #print(labels)

                    # Forward pass
                    outputs = self(images)
                    labels, _ = torch.max(labels, 1)
                    loss = criterion(outputs, labels)

                    # Obliczanie dokładności
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    accuracy = correct / labels.size(0)

                    # Zapisz stratę i dokładność
                    losses.append(loss.item())
                    accuracies.append(accuracy)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                    optimizer.step()

                    if (i + 1) % 1 == 0:
                        log_message = (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                                       f'Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}')
                        print(log_message)
                        f.write(log_message + '\n')



    def save_plots():
        # Rysowanie wykresów
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Accuracy')
        plt.legend()
        plt.show()


    def _initialize_weights(self):
        print('Inicjalizacja wag')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)