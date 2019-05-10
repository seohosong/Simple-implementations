import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size',  type=int, default=128)
    parser.add_argument('-epochs',  type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, num_workers=4)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(train_loader):
            model.train()

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print('Step %d Loss %0.4f' %(i+1, loss))

        with torch.no_grad():
            model.eval()
            correct = 0

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)

            print('Epoch %d Accuracy %0.5f' %(epoch+1, accuracy))


if __name__ == '__main__':
    main()
