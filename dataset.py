import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

np.set_printoptions(threshold=np.inf)

image_path = "data/images"
mask_path = "data/trimaps"
test_image_dir = "data/test_images"
test_mask_dir = "data/test_masks"

if not os.path.exists(test_image_dir):
    os.makedirs(test_image_dir)
if not os.path.exists(test_mask_dir):
    os.makedirs(test_mask_dir)

# parameters
test_size = 10
BATCH_SIZE = 16

# 手動指定訓練和測試數據集文件名
train_images = os.listdir(image_path)
train_masks = os.listdir(mask_path)
test_images = os.listdir(test_image_dir)
test_masks = os.listdir(test_mask_dir)

train_images.sort()
train_masks.sort()
test_images.sort()
test_masks.sort()

class CustomDataset(Dataset):
    def __init__(self, image_list, mask_list, image_dir, mask_dir, transform=None):
        self.image_list = image_list  # 圖片文件名列表
        self.mask_list = mask_list  # 遮罩文件名列表
        self.image_dir = image_dir  # 圖片資料夾路徑
        self.mask_dir = mask_dir  # 遮罩資料夾路徑
        self.transform = transform  # 圖片變換

    def __len__(self):
        return len(self.image_list)  # 返回數據集大小

    def __getitem__(self, idx):
        # 獲取圖片和遮罩文件名
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_list[idx])
        
        # 讀取圖片和遮罩
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)

        # 如果讀取圖像失敗，跳過這個樣本
        if image is None or mask is None:
            # print(f"Failed to load image or mask: {img_name}, {mask_name}")
            return self.__getitem__((idx + 1) % len(self.image_list))  # 嘗試下一個樣本
        
        # 處理遮罩
        mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
        
        # 調整圖片和遮罩大小
        imgsize = 256
        image = cv2.resize(image, (imgsize, imgsize))
        mask = cv2.resize(mask, (imgsize, imgsize))
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[-1]
        
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # 應用變換
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # 將遮罩二值化
        mask = (mask > 0.5).float()

        return image, mask  # 返回圖片和遮罩

# 定義圖片變換
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 調整圖片大小
    transforms.ToTensor()  # 轉換為Tensor
])

# 創建訓練和測試數據集
train_dataset = CustomDataset(train_images, train_masks, image_path, mask_path, transform=transform)
test_dataset = CustomDataset(test_images, test_masks, test_image_dir, test_mask_dir, transform=transform)

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 打印一張遮罩的numpy數據
# for _, mask in train_loader:
#     mask_np = mask[0].numpy()
#     print(mask_np)
#     break
