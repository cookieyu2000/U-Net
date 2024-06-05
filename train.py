# train.py
import torch
import torch.nn as nn
from unet import UNet
from dataset import train_loader
from tqdm import tqdm
import os
from torchsummary import summary
import matplotlib.pyplot as plt

# 定義路徑
weights_path = 'weights'

# 創建目錄
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超參數
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# 定义Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        logits = torch.sigmoid(logits)
        true = true.float()
        intersection = (logits * true).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + true.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return bce + dice



# 初始化模型、損失函數和優化器
model = UNet(in_channels=3, num_classes=1).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    model.train()
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, masks) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_description(f'Epoch: {epoch + 1}')

        avg_train_loss = train_loss / len(train_loader)

        print('='*30)
        print('Epoch: {}, Average Loss: {:.4f}'.format(epoch + 1, avg_train_loss))
        print('='*30)
            
        if avg_train_loss < best_loss:
            print(f'Train Loss: {avg_train_loss:.4f}, Loss improved from {best_loss:.4f} to {avg_train_loss:.4f}. Saving model...')
            best_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(weights_path, 'best_model.pth'))
        else:
            print(f'Train Loss: {avg_train_loss:.4f}, Loss did not improve from {best_loss:.4f}.')

if __name__ == "__main__":
    summary(model, (3, 256, 256))
    train()
