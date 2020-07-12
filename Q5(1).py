import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform1 = transforms.Compose([transforms.ToTensor(), normalize])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform1, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform1, download=False)


train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1=nn.Conv2d(1 ,10,5) 
        self.pool = nn.MaxPool2d(2,2) 
        self.conv2=nn.Conv2d(10,20,3) 
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)

    
    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x) 
        out = F.relu(out)
        out = self.pool(out)  
        out = self.conv2(out) 
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out
 


model = SimpleNet()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.02)

epoch = 0
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        
        images = Variable(images)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        


    correct1 = 0
    for images, labels in tqdm(test_loader):
            
        images = Variable(images)
        labels = Variable(labels)            
        outputs = model(images)
                
        _, predicted = torch.max(outputs.data, 1)

        correct= (predicted == labels).sum()
        correct1 +=correct.item()
            
    accuracy1 = 100 * correct1 / (len(test_dataset))

    print(accuracy1)

    
    
    correct2 = 0
    for images, labels in tqdm(train_loader):
            
        images = Variable(images)
        labels = Variable(labels)            
        outputs = model(images)
                
        _, predicted = torch.max(outputs.data, 1)

        correct= (predicted == labels).sum()
        correct2 +=correct.item()
            
    accuracy2 = 100 * correct2 / (len(train_dataset))
    print(accuracy2)


            
