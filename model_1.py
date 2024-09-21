import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms


from mec import *
from generalization import *
class VGG11_configuration_A(nn.Module):
    def __init__(self,in_channels,num_clasees=10,hidden_states=4096):
        super(VGG11_configuration_A,self).__init__()
        self.in_channels = in_channels
        self.num_clasees = num_clasees 
        self.hidden_states = hidden_states
        
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
            nn.Linear(512*1*1,self.hidden_states),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_states,self.hidden_states),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_states,self.num_clasees)
        )
        
    def forward(self,x):
        #(batch_size,3,32,32) --> (batch_size,512,1,1)
        conv = self.conv_layers(x)
        #(batch_size,512,1,1) --> (batch_size,10)
        conv = conv.view(conv.size(0),-1)
        conv = self.linear_layers(conv)
        return conv 

def train_test_epochs(model, train_loader, test_loader, epochs, lr, device, warmup_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0) 
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    model = model.to(device)
    model.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Initial testing"):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = loss_fn(output, label)
            initial_test_loss += loss.item()
    initial_test_loss /= len(test_loader)
    test_losses.append(initial_test_loss)

    for epoch in range(epochs):
        if epoch == warmup_epochs:  
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        model.train()
        for data, label in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            train_losses.append(loss.item())

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, label in tqdm(test_loader, desc=f"Testing epoch {epoch+1}/{epochs}"):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = loss_fn(output, label)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
    return train_losses, test_losses

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

def get_predicted(model,data_loader,device):
    model.eval()
    predicted_all = []
    with torch.no_grad():
        for data,label in tqdm(data_loader,desc="Accuracy"):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _,predicted = torch.max(output.data,1)
            predicted_all.append(predicted)
            # total += label.size(0)
            # correct += (predicted == label).sum().item()
    
    return predicted_all

def count_correct_predictions(predicted, actual):
    # Ensure both arrays are numpy arrays
    predicted = np.array(predicted)
    actual = np.array(actual)
    
    # Check where predictions match actual classes
    correct_predictions = predicted == actual
    
    # Initialize a dictionary to hold count of correct predictions for each class
    correct_count_per_class = {}
    
    # Iterate over each unique class in the actual labels
    for class_label in np.unique(actual):
        # Count correct predictions for the current class
        # We count True values in correct_predictions where the actual class is class_label
        correct_count_per_class[class_label] = np.sum(correct_predictions[actual == class_label])
    
    return correct_count_per_class

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(model)
    transform = transforms.Compose([transforms.Pad(2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Calculate max information in train data
    img_tensor = trainset.data     # shape 28 x 28 x 1 (padding has no information b.c. all 0s)
    mec_train = get_mec_data(img_tensor)
    print(f"train data contains maximally {mec_train} bits of information")
    
    num_hidden_layers = [8,16,32,64,128,256,512,1024,2048,4096]
    train_accuracy_all = []
    test_accuracy_all = []
    mec_ffnn_all = []
    generalization_all = []
    for hidden_layers in num_hidden_layers:
        model = VGG11_configuration_A(in_channels=1,num_clasees=10,hidden_states=hidden_layers)
        train_losses, test_losses = train_test_epochs(model, trainloader, testloader, epochs=5, lr=0.001, device=device)
        mec_ffnn = get_mec_ffnn(model.linear_layers)
        mec_ffnn_all.append(mec_ffnn)
        # Get compression ratio
        dim_in_conv =  model.in_channels * 32 * 32          # 1 x 32 x 32
        dim_out_conv = model.linear_layers[0].in_features    # 512 x 1 x 1
        comp_ratio = get_compression(dim_in_conv, dim_out_conv)
        print(f"Train data information content after compression: {mec_train//comp_ratio}")
        train_accuracy = accuracy(model,trainloader,device)
        train_accuracy_all.append(train_accuracy)
        test_accuracy = accuracy(model,testloader,device)
        test_accuracy_all.append(test_accuracy)
        predicted,labels = get_predicted_and_actuals(model,trainloader,device)
        p_dict = get_class_probabilities(labels)
        ps = [p_dict[i] for i in p_dict.keys()]

        # Get ks
        pred_correct_dict = count_correct_predictions(predicted, labels)
        ks = [pred_correct_dict[i] for i in pred_correct_dict.keys()]

        G = get_G(mec_ffnn, ks, ps)
        generalization_all.append(G)        


    with open('model_1_metrics_extra.txt', 'w') as f:
        f.write(f"Num hidden layers: {num_hidden_layers}\n")
        f.write(f"Train accuracy: {train_accuracy_all}\n")
        f.write(f"Test accuracy: {test_accuracy_all}\n")
        f.write(f"Mec ffnn: {mec_ffnn_all}\n")
        f.write(f"Generalization: {generalization_all}\n")