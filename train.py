import torch
import torch.nn as nn 
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import final_model
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(model, loader, loss_fn, tb_writer, device, epoch_index):
    running_loss = 0.
    total_loss = 0.
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return total_loss / len(loader)   

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
lr = 1e-4 
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])
train_data = datasets.CIFAR10(root = "/home/cj/models/quantization_project",train = True, transform = train_transform, 
                        download = True)
test_dataset  = datasets.CIFAR10(root="/home/cj/models/quantization_project", train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data,batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
loss_fn = torch.nn.CrossEntropyLoss()
model = final_model(dims = 96,depths= [1,2], mlp_ratio= 0.6, window_size= [8, 4], num_classes= 10, drop_rate= 0.6, drop_path_rate= 0.4, attn_drop_rate= 0.6, num_heads= [6,6]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/CIFAR_Mamba_trainer_{timestamp}')
epoch_number = 0 
best_vloss = 1_000_000.
checkpoint = torch.load('/home/cj/models/quantization_project/checkpoint_396.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_number = checkpoint['epoch']
for epoch in range(epochs):
    print(f'EPOCH {epoch_number + 1}:')

    model.train(True)
    
    avg_loss = train_one_epoch(epoch_index= epoch_number,loss_fn = loss_fn,model=model, loader= train_loader ,device = device,tb_writer = writer)


    running_vloss = 0.0
    
    checkpoint = {
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss
    }
    
    model.eval()
    
   
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            vinputs= vinputs.to(device)
            vlabels= vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(test_loader)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'checkpoint_{epoch_number +1 }.pth'
        torch.save(checkpoint , model_path)

    epoch_number += 1