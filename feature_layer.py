import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import os

def save_feature_map(features, layer_name, save_dir):
    try:
        n_features = features.shape[1]
        n_features_to_show = min(64, n_features)
        grid_size = int(np.ceil(np.sqrt(n_features_to_show)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        for i in range(grid_size * grid_size):
            ax = axes[i // grid_size, i % grid_size]
            if i < n_features_to_show:
                ax.imshow(features[0, i].detach().cpu().numpy(), cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')

        plt.savefig(os.path.join(save_dir, f'{layer_name}.jpg'), bbox_inches='tight')
        plt.close(fig)
        print(f"Feature map for {layer_name} saved successfully.")
    except Exception as e:
        print(f"Error saving feature map for {layer_name}: {e}")

def layer_outputs(image_path, weights_path, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    
    model = UNet(in_channels=3, num_classes=1).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).float().to(device)
        print("Image loaded and transformed successfully.")
    except Exception as e:
        print(f"Error loading or transforming image: {e}")
        return

    try:
        with torch.no_grad():
            out, layer_outputs = model(input_image)
        print("Forward pass completed successfully.")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        return
    
    layers = ['down_1', 'down_2', 'down_3', 'down_4', 'bottle_neck', 'up_1', 'up_2', 'up_3', 'up_4']
    
    try:
        for layer_name, features in zip(layers, layer_outputs):
            print(f"Processing layer: {layer_name}")
            save_feature_map(features, layer_name, save_dir)
        print("Feature maps saved successfully.")
    except Exception as e:
        print(f"Error processing or saving feature maps: {e}")
        return

    try:
        # Save final output prediction
        pred_mask = torch.sigmoid(out)
        pred_mask = (pred_mask > 0.5).float()
        pred_mask = pred_mask.squeeze().cpu().detach().numpy()
        
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'predict.jpg'), bbox_inches='tight')
        plt.close()
        print("Final prediction mask saved successfully.")
    except Exception as e:
        print(f"Error saving final prediction mask: {e}")

if __name__ == "__main__":
    weights_path = 'weights/best_model.pth'
    single_image_path = 'data/test_images/img1.jpg'
    save_dir = 'layer_outputs'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer_outputs(single_image_path, weights_path, save_dir, device)
