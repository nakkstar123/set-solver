from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from time import time

# -------------------------------------------------------------------------------------------------
# Dataset Loading
# -------------------------------------------------------------------------------------------------

class SetCardDataset(Dataset):
    """
    Recursively loads images from a directory. The leaf folder (the folder containing the image)
    is used as the class label. Assumes that the Google Drive folder follows the structure:

    root/
      ├── 0
      │   ├── 00
      │   │   ├── 000   <-- class label: "000"
      │   │   │   ├── IMG_5821.HEIC
      │   │   │   └── ...
      │   │   └── ...
      │   └── ...
      └── 1
          └── ...

    This example will automatically generate 81 classes.
    """
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of tuples: (image_path, label_str)
        self.downsample = transforms.Resize((img_size, img_size))

        # Recursively walk the directory structure
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    # Use the immediate parent folder as the label (leaf directory)
                    label_str = os.path.basename(root)
                    self.samples.append((file_path, label_str))

        # Create a sorted list of unique class labels and a mapping to index
        self.classes = sorted(list(set(label for _, label in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Convert string labels to integer indices
        self.samples = [(self._get_image(path), self.class_to_idx[label]) for path, label in tqdm(self.samples)]

    def _get_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.downsample(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            image = self.transform(img)
        return image, label
    

def load_dataset(root_dir, img_size=224):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SetCardDataset(root_dir, transform=test_transform, img_size=img_size)
    return dataset

def load_dataloader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# -------------------------------------------------------------------------------------------------
# Model Testing Helper Functions
# -------------------------------------------------------------------------------------------------

def validate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    time = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            start_time = time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            end_time = time()
            time.append(end_time - start_time)

    return total, correct, time

def model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.nelement() for p in model.parameters())

    # Convert to MB
    total_size = total_size / (1024 * 1024)

    return total_params, total_size

def model_jit_optimize(model):
    return torch.jit.script(model)

# -------------------------------------------------------------------------------------------------
# Model Loading, and Testing
# -------------------------------------------------------------------------------------------------

def load_model(model_path, device):
    model = torch.load(model_path)
    model.to(device)
    return model

def test_model(model, dataloader, device):
    total, correct = validate_model(model, dataloader, device)
    return total, correct

def model_mapping(model_name):
    if model_name == "v2_3":
        return "model/weights/finetuned_mobilenetv2_check1_3feature.pth"
    elif model_name == "v2_7":
        return "model/weights/finetuned_mobilenetv2_check1_7feature.pth"
    elif model_name == "v2_full":
        return "model/weights/finetuned_mobilenetv2_check1_full.pth"
    else:
        return "model/weights/finetuned_mobilenetv3_large_check1_full.pth"
    
def report_model_results(model_name, dataloader, device):
    model_path = model_mapping(model_name)
    model = load_model(model_path, device)
    total, correct = test_model(model, dataloader, device)
    print(f"Model: {model_name}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print(f"Model Size: {model_info(model)[1]:.2f} MB")
    print(f"Model Params: {model_info(model)[0]}")
    print(f"Total Time: {sum(time):.2f} seconds")
    print(f"Max Inference Time: {max(time):.2f} seconds")
    print(f"Average Inference Time: {sum(time) / len(time):.2f} seconds")
    print(f"Throughput: {len(time) / sum(time):.2f} images/second")

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    for model in ["v2_3", "v2_7", "v2_full", "v3_full"]:
        report_model_results(model, load_dataloader(load_dataset("data/")), device)