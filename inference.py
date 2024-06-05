import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from dataset import CustomDataset
from unet import UNet

np.set_printoptions(threshold=np.inf)

def pred_show_img_grid(data_path, weights_path, device):
    # Load the model
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    image_dataset = CustomDataset(data_path, transform=transforms.ToTensor())
    images = []
    original_masks_list = []
    pred_masks_list = []

    for img, original_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid to get probabilities
            pred_mask = (pred_mask > 0.5).float()  # Binarize the output

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()

        original_mask = original_mask.cpu().detach()

        images.append(img)
        original_masks_list.append(original_mask)
        pred_masks_list.append(pred_mask)

    images.extend(original_masks_list)
    images.extend(pred_masks_list)
    fig = plt.figure(figsize=(12, 8))
    for i in range(1, 3*len(images)+1):
        ax = fig.add_subplot(3, len(images)//3, i)
        ax.imshow(images[i-1], cmap='gray')
        ax.axis('off')
    plt.show()

def single_image_inference(image_path, weights_path, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_mask = model(input_image)
        if isinstance(pred_mask, tuple):
            pred_mask = pred_mask[0]  # Assuming the first element is the mask
        pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid to get probabilities
        pred_mask = (pred_mask > 0.5).float()  # Binarize the output

    input_image = input_image.squeeze(0).cpu().detach()
    input_image = input_image.permute(1, 2, 0)
    
    pred_mask = pred_mask.squeeze().cpu().detach()
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    plt.savefig('output/results_1.png', bbox_inches='tight')

    # 输出预测掩码的一些统计信息
    # print("Predicted Mask - Unique values:", np.unique(pred_mask.numpy()))

if __name__ == "__main__":
    weights_path = 'weights/best_model.pth'
    single_image_path = 'data/test_images/img1.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    single_image_inference(single_image_path, weights_path, device)
