import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.relu(x)

class reshape(nn.Module):
    def __init__(self, row, col):
        super(reshape, self).__init__()
        self._row = row
        self._col = col

    def forward(self, x):
        return x.view(self._row, self._col)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.backends.cudnn.enabled
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download = True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=100, shuffle=False)

    input_size = 784
    hidden_size = 500
    output_size = 10
    num_epochs = 10
    learning_rate = 0.001

    model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(),
            reshape(100, -1),
            nn.Linear(320, 100),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.ReLU()
        )
    model = model.to(device)
    print(model)

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = lossFunction(out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # fine tunning
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.01)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = lossFunction(out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # accuracy
    corr = 0.;
    total = 0.;
    for j, (test_images, test_labels) in enumerate(test_loader):
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        valid = model(test_images)
        corr += (valid.argmax(dim=1)==test_labels).sum().item()
        total += len(test_labels)

    accuracy = 100.00 * corr / total
    print('Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, total, accuracy))
    #print('Epoch [{}/{}], Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(epoch+1, num_epochs, corr, total, accuracy))


if __name__ == '__main__':
    main()


