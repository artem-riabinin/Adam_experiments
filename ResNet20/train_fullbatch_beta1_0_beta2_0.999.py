import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



# wandb logging
wandb_log = True 
wandb_project = 'resnet20'
wandb_run_name = 'resnet20_fullbatch_beta1_0_beta2_0.999'



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Hyper-parameters
num_epochs = 300
learning_rate = 0.001
beta1 = 0
beta2 = 0.999
betas = (beta1, beta2)
mini_batch_size = 10000
accumulation_steps = 5
batch_size = mini_batch_size * accumulation_steps



# wandb logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
configuration = {k: globals()[k] for k in config_keys}
if wandb_log:
    run = wandb.init(project=wandb_project, name=wandb_run_name, config=configuration)



# Image preprocessing modules
normalize = transforms.Normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

train_dataset = torchvision.datasets.CIFAR10(root='data/',train=True, transform=train_transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data/',train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=mini_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=mini_batch_size, shuffle=False)



# Basic Architecture for ResNet
def conv3x3(in_channels, out_channels, stride=1):
    """
    return 3x3 Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Initialize basic ResidualBlock with forward propogation
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Initialize  ResNet with forward propogation
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



# Initialize ResNet20
model = ResNet(ResidualBlock, [3, 3, 3]).to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)



# Train the model
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_train_loss += loss.item()
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Average train loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)

    # Test evaluation
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

            # Predictions and accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = running_test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, '
          f'Test Accuracy: {accuracy:.2f}%')
      
    if wandb_log:
        wandb.log({
            "epoch": epoch+1,
            "train/loss": avg_train_loss,
            "test/loss": avg_test_loss,
            "test/accuracy": accuracy,
        })
        
run.finish()