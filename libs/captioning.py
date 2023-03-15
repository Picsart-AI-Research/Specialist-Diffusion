import os
import torch
import argparse
from PIL import Image
from models.blip import blip_decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

IMAGE_SIZE = 384


def load_image(raw_image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).cuda()
    return image


if __name__ == '__main__':
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', type=str, help='The path to the dataset folder')
    args = parser.parse_args()

    model = blip_decoder(pretrained=model_url, image_size=IMAGE_SIZE, vit='base')
    model.eval()
    model = model.cuda()

    files = os.listdir(args.data)
    source_list = dict()

    with torch.no_grad():
        for file in files:
            image = load_image(Image.open(os.path.join(args.data, file)).convert('RGB'))
            caption = model.generate(image, sample=True, num_beams=5, max_length=20, min_length=5)
            source_list[file] = caption[0]
            print(file, ' caption: ' + caption[0])

    json.dump(source_list, open('{}/source_list.json'.format(args.data), 'w'), indent=4)
