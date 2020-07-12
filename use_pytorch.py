import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch import nn

class Simple_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Simple_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


BATCH_SIZE = 64 #change default value
NUM_EPOCHS = 20 #change default value

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

model = Simple_Net(28 * 28, 300, 100, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

epoch = 0
for epoch in range(NUM_EPOCHS):
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
train_acc = 0
test_acc = 0
for data in train_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    out = model(img)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    train_acc += num_correct.item()

for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    out = model(img)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    test_acc += num_correct.item()

print(' train_acc: {:.6f},test_acc: {:.6f}'.format(
    train_acc / (len(train_dataset)),test_acc / (len(test_dataset))
))
