import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class NeuralNet(nn.Module):
    """A Neural Network with a hidden layer"""
    def __init__(self, input_size,hidden_size,output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        return self.relu(output)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.backends.cudnn.enabled
    
    train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  num_workers=8, batch_size=100, shuffle=False)

    input_size = 784
    hidden_size = 500
    output_size = 10
    num_epochs = 5
    learning_rate = 0.001

    model = NeuralNet(input_size, hidden_size, output_size)
    model = model.to(device)
    print(model)

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)


    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            images = images.reshape(-1, 28*28)

            out = model(images)
            loss = lossFunction(out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            accuracy = (out.argmax(dim=1) == labels).sum() * 100. / len(labels)

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.2f} %' .format(epoch+1, num_epochs, i+1, total_step, accuracy))


if __name__ == '__main__':
    main()


