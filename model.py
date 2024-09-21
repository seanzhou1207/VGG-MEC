import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms


class VGG11_configuration_A(nn.Module):
    def __init__(self,in_channels,num_clasees=10):
        super(VGG11_configuration_A,self).__init__()
        self.in_channels = in_channels
        self.num_clasees = num_clasees 
        
        self.conv_layers = nn.Sequential(
            #(batch_size,3,32,32) --> (batch_size,64,32,32)
            nn.Conv2d(in_channels=self.in_channels,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,64,32,32) --> (batch_size,64,16,16) 
            nn.MaxPool2d(kernel_size=2,stride=2),
            #(batch_size,64,16,16) --> (batch_size,128,16,16)
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,128,16,16) --> (batch_size,128,8,8)
            nn.MaxPool2d(kernel_size=2,stride=2),
            #(batch_size,128,8,8) --> (batch_size,256,8,8)
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,256,8,8) --> (batch_size,256,4,4)
            nn.MaxPool2d(kernel_size=2,stride=2),
            #(batch_size,256,4,4) --> (batch_size,512,4,4)
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,512,4,4) --> (batch_size,512,4,4)
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,512,4,4) --> (batch_size,512,2,2)
            nn.MaxPool2d(kernel_size=2,stride=2),
            #(batch_size,512,2,2) --> (batch_size,512,2,2)
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,512,2,2) --> (batch_size,512,2,2)
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            #(batch_size,512,2,2) --> (batch_size,512,1,1)
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(4096,self.num_clasees)
        )
        
    def forward(self,x):
        #(batch_size,3,32,32) --> (batch_size,512,1,1)
        conv = self.conv_layers(x)
        #(batch_size,512,1,1) --> (batch_size,10)
        conv = conv.view(conv.size(0),-1)
        conv = self.linear_layers(conv)
        return conv 




def train_test_epochs(model,train_loader,test_loader,epochs,lr,device):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    model = model.to(device)
    model.eval()
    inital_test_loss = 0.0
    with torch.no_grad():
        for data,label in tqdm(test_loader,desc="Iniital testing"):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_fn(output,label)
            inital_test_loss += loss.item()
    inital_test_loss /= len(test_loader)
    test_losses.append(inital_test_loss)
    
    for epoch in range(epochs):
        model.train()
        for data,label in tqdm(train_loader,desc=f"Training epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data,label in tqdm(test_loader,desc=f"Testing epoch {epoch+1}/{epochs}"):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = loss_fn(output,label)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
    return train_losses,test_losses

def accuracy(model,data_loader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data,label in tqdm(data_loader,desc="Accuracy"):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _,predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return (correct/total)*100

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11_configuration_A(in_channels=1,num_clasees=10)
    #print(model)
    transform = transforms.Compose([transforms.Pad(2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    train_losses, test_losses = train_test_epochs(model, trainloader, testloader, epochs=5, lr=0.001, device=device)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved successfully")
    print(f"Train losses: {train_losses}")
    print(f"Test losses: {test_losses}")
    train_accuracy = accuracy(model, trainloader, device)
    test_accuracy = accuracy(model, testloader, device)
    print(f"Train accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
