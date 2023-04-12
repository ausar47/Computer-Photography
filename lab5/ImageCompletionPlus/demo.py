import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torchvision.transforms.functional as torchvision_F
import network


def postprocess(img):
    img = (img + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img


def demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')    # GPU 8GB 显存不够用

    model = network.Generator().to(device)
    path = 'model/facade/model.pth'
    path = 'model/dtd/model.pth'
    path = 'model/celebahq/model.pth'
    path = 'model/places2/model.pth'
    path = 'model/places2/model1.pth'
    model.load_state_dict(torch.load(path, map_location=device)['netG'])
    model.eval()

    image = Image.open('images/case4_input.png').convert('RGB')
    mask = Image.open('images/case4_mask.png').convert('L')

    w, h = image.size
    divisor = 64
    h = (h | (divisor - 1)) + 1
    w = (w | (divisor - 1)) + 1
    image = image.resize((w, h))
    mask = mask.resize((w, h))

    image = torchvision_F.to_tensor(image) * 2 - 1
    mask = torchvision_F.to_tensor(mask)
    image_masked = image * (1 - mask) + mask

    images_masked = image_masked[None].to(device)
    masks = mask[None].to(device)
    with torch.no_grad():
        output = model(torch.cat((images_masked, masks), dim=1), masks)

    img = postprocess(output)[0]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    demo()
