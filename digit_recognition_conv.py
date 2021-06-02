import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

# Neural Network with CNN + pooling layers
# note: dropout layers are disabled, please check nn.Dropout2d() for information
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        #x = self.relu(self.maxpool2(self.conv2_drop(self.conv2(x))))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        #x = self.conv2_drop(x)
        x = self.fc2(x)
        return self.relu(x)

def main():

    # check cpu/gpu usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #torch.backends.cudnn.enabled
    
    # training dataset + dataloader
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=1000, shuffle=True)

    # validation dataset + dataloader
    valid_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download = True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, num_workers=4, batch_size=1000, shuffle=False)

    # hyper parameters
    input_size = 784
    hidden_size = 500
    output_size = 10
    num_epochs = 5
    learning_rate = 0.001

    # initiate a neural network
    model = Net()
    model = model.to(device)
    print(model)

    # loss function
    lossFunction = nn.CrossEntropyLoss()

    # learnable parameters optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)

    # loop over epochs
    for epoch in range(num_epochs):

        # loop over training dataset
        for i, (images,labels) in enumerate(train_loader):

            # move data from main memory to device (cpu/gpu) associated memory
            images = images.to(device)
            labels = labels.to(device)

            # output and loss
            out = model(images)
            loss = lossFunction(out,labels)

            # adjust learnable parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate accuracy with validation dataset
        corr = 0.;
        total = 0.;
        for j, (valid_images, valid_labels) in enumerate(valid_loader):
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)

            out = model(valid_images)
            corr += (out.argmax(dim=1)==valid_labels).sum().item()
            total += len(valid_labels)

        accuracy = 100.00 * corr / total
        print('Epoch [{}/{}], Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(epoch+1, num_epochs, corr, total, accuracy))


    # saving model
    model_path = './digit_recognizer_cnn.pt'
    torch.save(model.state_dict(), model_path)

    # initiate a new neural network
    model_valid = Net()

    # load a well-trained network
    model_valid.load_state_dict(torch.load(model_path, map_location='cpu'))

    # use the well-trained network and calculate accuracy
    corr=0
    total=0
    for j, (valid_images, valid_labels) in enumerate(valid_loader):
        valid_images = valid_images.to(device)
        valid_labels = valid_labels.to(device)

        out = model_valid(valid_images)
        corr += (out.argmax(dim=1)==valid_labels).sum().item()
        total += len(valid_labels)

    accuracy = 100.00 * corr / total
    print('Validation Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, total, accuracy))


if __name__ == '__main__':
    main()


