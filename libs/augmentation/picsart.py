import os
import json
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T


prompt_mixer_list = [    
    "{item} in the style of {style}",
    "{item} in {style} style",
    "a picture of {item} in {style} style",
    "a rendering of {item} in {style} style",
    "a painting of {item} in {style} style",    
]


synonyms = {
    "boy": ['kid', 'youngster', 'junior'],
    "hairs": ['locks', 'tresses', 'mop'],
    "hair": ['lock', 'tress', 'mop'],
    "eyes": ['eyeballs', 'peeper'],
    "woman": ['lady', 'girl', 'femme', 'maiden'],
    "finger": ['thumb', 'forefinger', 'digit'],
    "branch": ['bough', 'shoot', 'spray'],
    "branches": ['boughs', 'shoots', 'sprays'],
    "flowers": ['blooms', 'blossom', 'efflorescence'],
    "flower": ['bloom', 'blossom', 'efflorescence'],
    "leaves": ['fronds', 'cotyledons'],
    "leaf": ['frond', 'cotyledon'],
    "hand": ['palm', 'fist', 'paw'],
    "hands": ['palms', 'fists', 'paws'],
    "face": ['countenance', 'visage', 'features'],
    "horse": ['nag', 'mare', 'colt', 'pony'],
    "tail": ['brush', 'extremity', 'appendage'],
    "stem": ['stalk', 'branch', 'trunk', 'stock'],
    "stems": ['stalks', 'branches', 'trunks', 'stocks'],
    "cat": ['feline', 'kitty', 'tabby'],
    "air": ['wind', 'blast', 'breeze', 'puff'],
    "wings": ['pinion', 'pennon', 'arm'],
    "pointer": ['hint', 'tip', 'signal'],
    "rabbit": ['buck', 'doe'],
    "legs": ['shanks', 'limbs', 'stumps'],
    "heart": ['feeling', 'sentiment', 'affection'],
    "balloon": ['airship', 'hot-air balloon', 'Montgolfier'],
    "head": ['skull', 'crown', 'pate', 'cranium'],
    "fox": ['reynard'],
    "body": ['physique', 'figure', 'shape', 'build'],
    "dress": ['frock', 'shift', 'garment', 'robe'],
    "foot": ['paw', 'pad', 'trotter'],
    "rose": ['flower'],
    "sneakers": ['footwear', 'shoe', 'footgear'],
    "Earth": ['planet', 'world', 'globe', 'sphere', 'orb'],
    "bush": ['shrub', 'plant', 'hedge', 'shrubbery'],
    "lemon": ['citric', 'citrine', 'citrous'],
    "petal": ['frond', 'needle', 'stalk', 'blade'],
    "petals": ['fronds', 'needles', 'stalks', 'blades'],

    "beautiful": ['nice', 'pretty', 'attractive', 'lovely'],
    "long": ['elongated', 'extended', 'lengthy'],
    "mindful": ['apprehensive', 'sensible', 'attentive', 'careful'],
    "angry": ['enraged', 'furious', 'incensed', 'inflamed'],
    "many": ['multifold', 'multiple', 'numerous', 'several', 'a couple of', 'myriad'],
    "young": ['youthful', 'juvenile', 'immature'],
    "near": ['around', 'next to', 'nigh', 'alongside'],
    "small": ['little', 'pint-size', 'slight', 'smallish', 'toylike'],
    "cute": ['beguiling', 'nice', 'beautiful'],
    "half": ['division', 'fraction', 'part', 'segment', 'section'],
    "surprised": ['amazed', 'astonished', 'wondering'],
    "black": ['dark', 'inky', 'brunet'],
    "modest": ['common', 'standard', 'usual', 'regular'],
    "curly": ['crimped', 'frizzy', 'wavy'],
    "comfortable": ['common', 'easeful', 'cozy', 'relaxing'],
    "main": ['central', 'dominant', 'grand', 'prime', 'leading'],
    "multiple": ['many', 'numerous', 'several', 'myriad'],
    "big": ['fatal', 'great', 'heavy', 'serious', 'weighty'],
    "whole": ['all', 'entire', 'undivided', 'complete'],

    # flat design
    "red": ['ruddy', 'crimson', 'cherry', 'carmine', 'ruby'],
    "coffin": ['box', 'casket', 'sarcophagus'],
    "white": ['light', 'blanched'],
    "purple": ['violet', 'purplish'],
    "stars": ['celestial bodies', 'starlets'],
    "corner": ['area', 'nook'],
    "dark": ['twilight', 'black', 'dimout', 'night', 'brownout'],
    "center": ['core', 'point', 'area'],
    "lines": ['marks', 'prints', 'edges'],
    "bat": ['eutherian mammal', 'Chiroptera'],
    "sad": ['unhappy', 'regretful', 'tragic', 'sorrowful'],
    "robot": ['mechanism', 'humanoid', 'automaton'],
    "green": ['chromatic', 'greenish'],
    "witch": ['enchantress', 'persecutor', 'tormentor'],
    "tone": ['style', 'color', 'plangency', 'tine'],
    "color": ['style', 'tone', 'plangency', 'tine'],
    "colors": ['styles', 'tones', 'plangency', 'tines'],
    "flag": ['sign', 'streamer', 'tricolour', 'emblem'],
    "handle": ['knob', 'hold', 'grip'],
    "holder": ['knob', 'hold', 'grip'],
    "eagle": ['raptor', 'bird', 'harpy'],
    "yellow": ['xanthous', 'maize', 'amber', 'gold'],
    "yellowish": ['xanthous', 'maize', 'amber', 'gold'],
    "beak": ['snout', 'nose', 'hooter'],
    "broom": ['cleaning equipment', 'besom', 'whisk'],
    "pink": ['coral', 'rosiness', 'carnation'],
    "coconut": ['copra', 'food', 'fruit'],
    "tree": ['bush', 'shrub', 'sapling'],
    "trees": ['bushes', 'shrubs', 'saplings'],
    "frame": ['photograph', 'picture', 'enclose', 'exposure'],
    "light": ['bright', 'shining', 'glare', 'glow'],
    "bright": ['light', 'shining', 'glare', 'glow'],
    "blue": ['azure', 'sapphire', 'navy', 'saxe'],
    "medicine": ['treatment', 'remedy', 'bill', 'cure'],
    "pot": ['container', 'box', 'saucepan', 'vessel'],
    "bubble": ['castle in the air', 'vanity'],
    "pumpkin": ['vegetable', 'veggie'],
    "smile": ['grin', 'beam', 'smirk', 'leer'],
    "temple": ['chapel', 'joss house', 'church', 'pantheon'],
    "fireworks": ['explosions', 'illuminations'],
    "text": ['sentence', 'string', 'wording', 'content'],
    "dots": ['spots', 'specks', 'points', 'marks'],
    "circle": ['wheel', 'round', 'spiral', 'orbit'],
    "seeds": ['semen', 'sperms', 'milts', 'emissions'],
    "goblet": ['glass', 'tass', 'cup', 'beaker'],
    "patterns": ['design', 'decorations', 'markings', 'ornaments'],
    "pattern": ['design', 'decoration', 'marking', 'ornament'],
    "candle": ['taper', 'sconce', 'cierge'],
    "picture": ['photograph', 'painting', 'print', 'canvas', 'drawing', 'sketch'],
    "apple": ['fruit', 'pome'],
    "banana": ['fruit', 'Musa acuminata', 'herb'],
    "grape": ['fruit', 'berry'],
    "pear": ['fruit', 'anjou', 'pome'],
    "gray": ['blackish', 'colourless', 'smoky', 'silver'],
    "present": ['gift', 'handout', 'bounty'],
    "box": ['pot', 'container', 'packet', 'bin'],
    "node": ['junction', 'fork', 'intersection', 'crossing'],
    "ear": ['piece', 'cereal'],
    "corn": ['grain', 'cereal'],
    "drum": ['canister', 'bin', 'can', 'cylinder'],
    "kerchief": ['bandanna', 'handkerchief', 'scarf', 'shawl']
}


def convert_image_to(image, img_type):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class PARandAugment(torch.nn.Module):
    def __init__(self, op_num=2, p=0.5):
        super().__init__()
        self.p = p
        self.op_num = op_num
        self.__ops = [
            T.RandomInvert(p=1.),
            T.RandomPosterize(bits=2, p=1.),
            T.RandomSolarize(threshold=100.0, p=1.),
            T.RandomAdjustSharpness(sharpness_factor=3, p=1.),
            T.RandomAutocontrast(p=1.),
            T.RandomEqualize(p=1.),
            T.RandomHorizontalFlip(p=1.),
            T.RandomVerticalFlip(p=1.)
        ]

    def forward(self, x):
        if np.random.rand() < self.p:
            transforms = np.random.choice(self.__ops, (self.op_num, ))
            aug_str = ' with the ' + transforms[0].__class__.__name__ + ' transformation'
        else:
            transforms = []
            aug_str = ''

        for t in transforms:
            x = t(x)

        return x, aug_str


class PicsartData(torch.utils.data.Dataset):
    def __init__(self, data_root, source_list, tokenizer, style_name, use_text_inversion, augmentation, image_size, clip_similars=None):
        super().__init__()
        self.data_root = data_root        
        self.data = json.load(open(source_list, 'r'))
        self.key_list = list(self.data.keys())
        self.tokenizer = tokenizer
        self.style_name = style_name
        self.use_text_inversion = use_text_inversion
        self.augmentation = augmentation
        self.image_size = image_size

        self.image_augmentation = PARandAugment(op_num=1)

        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(self.image_size),            
        ])

        self.similar_captions = torch.load(clip_similars, map_location='cpu')

    def __len__(self):
        return len(self.key_list)

    def synonym_replace(self, raw):
        word_list = raw.split()
        for i, word in enumerate(word_list):
            if word in synonyms.keys():
                word_list[i] = np.random.choice(synonyms[word] + [word])
        caption_str = ' '.join(word_list)
        return caption_str

    def __getitem__(self, idx):
        file_name = self.key_list[idx]
        image = Image.open(os.path.join(self.data_root, file_name))
        image = convert_image_to(image, 'RGB')
        raw_caption = self.data[file_name]                

        if len(self.augmentation) > 0:             
            if 'retrieval' in self.augmentation and file_name in self.similar_captions:
                similar_captions = self.similar_captions[file_name]
                randint = torch.randint(len(similar_captions), (1,))
                caption_str = similar_captions[randint]
            else:
                caption_str = self.data[file_name]
            
            if 'synonym' in self.augmentation:
                raw_caption = self.synonym_replace(raw_caption)
                caption_str = self.synonym_replace(caption_str)
        else:
            caption_str = raw_caption

        caption = random.choice(prompt_mixer_list).format(item=caption_str, style=self.style_name)

        if 'double' in self.augmentation:
            image, aug_name = self.image_augmentation(image)
            caption += aug_name

        image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        raw_ids = self.tokenizer(
            raw_caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example = {
            'pixel_values': image,
            'text': caption,
            'raw_caption': raw_caption,
            'image_path': os.path.join(self.data_root, self.key_list[idx]),
            'input_ids': input_ids,
            'raw_ids': raw_ids
        }

        return example