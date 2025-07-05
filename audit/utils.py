import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def save_imagefolder(dataset, target_path, maps=None):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]  
        if hasattr(dataset, 'classes'):
            class_name = dataset.classes[label]
        else:
            if maps is None:
                raise ValueError("Please provide a mapping of label to class name")
            class_name = maps[label]
        class_dir = os.path.join(target_path, class_name)
        os.makedirs(class_dir, exist_ok=True)  
        
        img = to_pil_image(img) if not isinstance(img, Image.Image) else img
        
        target_file = os.path.join(class_dir, f"{idx}.png")
        img.save(target_file)


def save_images_by_label(dataset, target_path, classes):
    # Create target path
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)

    # Define an image transformer to convert PyTorch tensors to PIL images

    for idx, (image, label) in enumerate(dataset):
        # Create a subfolder for each label
        label_path = os.path.join(target_path, classes[label])
        os.makedirs(label_path, exist_ok=True)

        # Convert PyTorch tensor to PIL image and save it
        img = to_pil_image(image)
        img.save(os.path.join(label_path, f"{idx}.png"))

