import torch
from torchvision.transforms import transforms

import config
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_image(path,device):
    img = Image.open(path)
    # img.show()
    transform = transforms.Compose([
                                    transforms.Resize(size=(224, 224)),
                                    # transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(
                                    # [0.34437596,0.38029084, 0.40777029],
                                    # [0.20266077, 0.13689659, 0.11555012]),
                                    ])
    
    img = transform(img)
    img = img.to(device)
    return img




def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)
    