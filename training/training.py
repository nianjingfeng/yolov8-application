import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import torchvision

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 3)
    def forward(self, x):
        x = self.model(x)
        return x
    
#early stopping
class early_stopping():
    def __init__(self,patience=5,verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    def __call__(self,val_loss,model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(model, data_loader, epochs=100, path = './model/best.pth'):
    #split the validation set from data_loader
    train_loader, val_loader = torch.utils.data.random_split(data_loader, [int(len(data_loader)*0.7),int(len(data_loader)*0.3)])
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_loader, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopper = early_stopping()
    best_acc = 0.0
    #train the model
    for epoch in tqdm(range(epochs),desc='epoch'):
        total_loss = 0.0
        for i, data in enumerate(tqdm(train_loader,desc='batch',leave=False)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        if (100 * correct / total )> best_acc:
            best_acc = 100 * correct / total
            torch.save(model.state_dict(), path)
        tqdm.write('-----------epoch: %d, loss: %.3f, val_acc: %.3f---------------' % (epoch + 1, total_loss, 100 * correct / total))
        early_stopper(total_loss,model)
        if early_stopper.early_stop:
            print('Early stopping')
            break

def test(model,path,dataloader):
    model.load_state_dict(torch.load(path))
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=32, shuffle=True)
    gt = []
    pred = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            gt.extend(labels)
            pred.extend(predicted)
    print(classification_report(gt, pred))
    print(confusion_matrix(gt, pred))

if __name__ == '__main__':
    model = resnet()
    data_loader = torchvision.datasets.ImageFolder(root='./data',transform=torchvision.transforms.ToTensor())
    train(model,data_loader,epochs=100,path='./model/best_resnet_mix.pth')
    test(model,'./model/best_resnet_mix.pth',data_loader)