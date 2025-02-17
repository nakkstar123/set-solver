from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, mobilenet_v3_large
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
from time import time
import matplotlib.pyplot as plt

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

        # Convert string labels to integer indices
        self.samples = [(self._get_image(path), self._convert_label(label)) for path, label in tqdm(self.samples)]

    def _convert_label(self, label):
        idx = 0
        for i, num in enumerate(label[::-1]):
            idx += int(num) * (3 ** i)
        return idx

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

def load_dataloader(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# -------------------------------------------------------------------------------------------------
# Model Testing Helper Functions
# -------------------------------------------------------------------------------------------------

def validate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    inf_time = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            start_time = time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).item()

            if outputs.argmax(dim=1).item() != labels.item():
                print(f"Incorrect: {outputs.argmax(dim=1).item()} != {labels.item()}")
                plt.imshow(images[0].permute(1, 2, 0))
                plt.show()
            end_time = time()
            inf_time.append(end_time - start_time)

    return total, correct, inf_time

def test_image(model, image, device):
    model.eval()
    image = image.to(device)
    output = model(image)
    return output.argmax(dim=1).item()

def load_image(path, img_size=224):
    downsample = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = downsample(Image.open(path).convert("RGB"))

    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    # Convert to 4D tensor
    img = img.unsqueeze(0)
    return img

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

def load_model(model_name, model_path, device):
    if model_name == "v3_full":
        model = mobilenet_v3_large()
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 81)
    else:
        model = mobilenet_v2()
        model.classifier[-1] = torch.nn.Linear(model.last_channel, 81)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device) 
    model = model_jit_optimize(model)
    model.eval()
    return model

def test_model(model, dataloader, device):
    total, correct, inf_time = validate_model(model, dataloader, device)
    return total, correct, inf_time

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
    model = load_model(model_name, model_path, device)
    total, correct, inf_time = test_model(model, dataloader, device)
    print(f"Model: {model_name}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
    print(f"Model Size: {model_info(model)[1]:.2f} MB")
    print(f"Model Params: {model_info(model)[0]}")
    print(f"Total Time: {sum(inf_time):.2f} seconds")
    print(f"Max Inference Time: {max(inf_time):.2f} seconds")
    print(f"Average Inference Time: {sum(inf_time) / len(inf_time):.2f} seconds")
    print(f"Throughput: {len(inf_time) / sum(inf_time):.2f} images/second")

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    dataloader = load_dataloader(load_dataset("model/data/outputs/"))
    for model in ["v3_full"]:
        report_model_results(model, dataloader, device)

    # model = load_model("v3_full", "model/weights/finetuned_mobilenetv3_large_check1_full.pth", device)
    # image = load_image("model/data/outputs/1111/IMG_6056_card_4.png")
    # res = test_image(model, image, device)
    # print(res)
