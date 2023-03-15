import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize(224),  # scale imported image
        transforms.ToTensor()])

    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(x):
    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def style_loss(f1, f2):
    with torch.no_grad():
        g1 = gram_matrix(f1)
        g2 = gram_matrix(f2)

    return F.mse_loss(g1, g2)


def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)
    return (x - mean) / std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str)
    parser.add_argument('--path2', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    vgg = vgg[:-2]

    images1 = list(Path(args.path1).glob('*.png'))
    images2 = list(Path(args.path2).glob('*.png'))

    style_losses = []
    for img1 in tqdm(images1):
        for img2 in images2:
            image1 = image_loader(str(img1))
            image2 = image_loader(str(img2))

            f1 = vgg(normalize(image1))
            f2 = vgg(normalize(image2))

            s_loss = style_loss(f1, f2)
            style_losses.append(s_loss)

    print('DB: {}, Avg Style Loss: {:.5}'.format(args.path1.split('/')[-1], np.mean(style_losses)))
    print('End')

