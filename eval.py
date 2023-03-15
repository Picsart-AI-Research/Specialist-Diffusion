import os
import json
import torch
import random
import argparse
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline
from libs.augmentation.picsart import prompt_mixer_list


def image_grid(imgs, rows, cols, size):
    assert len(imgs) <= rows*cols
        
    w, h = size, size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img.resize((w, h)), box=(i%cols*w, i//cols*h))
    return grid


def dummy_check(images, **kwargs):
    return images, False


def generate_with_pipe(prompt, num_samples, num_rows, generator, num_steps):        
    pipe.safety_checker = dummy_check
    all_images = [] 
    for _ in range(num_rows):
        with autocast("cuda"):
            images = pipe([prompt] * num_samples, num_inference_steps=num_steps, guidance_scale=7.5, generator=generator).images
            all_images.extend(images)
    
    return all_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', action='store', type=str, help='The path to the configuration file')
    parser.add_argument('--prompt', action='store', type=str, nargs='?', default='')
    parser.add_argument('--prompt-file', action='store', type=str, nargs='?', default='')
    args = parser.parse_args()
    
    config = json.load(open(args.config, 'r'))
    if args.prompt_file == '':
        if args.prompt == '':
            prompt = config['prompt']
        else:
            prompt = args.prompt    
        prompts = [prompt]
    else:
        prompts = open(args.prompt_file, 'r').readlines()

    target_dir = config['output_dir']
    if not os.path.exists(target_dir):    
        os.mkdir(target_dir)    

    num_samples = config['num_samples']
    num_rows = config['num_rows']
    num_cols = num_samples // num_rows

    generator = torch.Generator(device=torch.device('cuda'))
    generator.manual_seed(config['seed'])
    state = generator.get_state()

    for epoch in config['checkpoints']:
        generator.set_state(state)
        print('Generating results at epoch {}'.format(epoch))
        pipe = StableDiffusionPipeline.from_pretrained(
            os.path.join(config['model_dir'], 'epoch{}'.format(epoch)),
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.safety_checker = dummy_check

        for i, prompt in enumerate(prompts):
            if not os.path.exists(os.path.join(target_dir, str(i + 1))):
                os.mkdir(os.path.join(target_dir, str(i + 1)))
                        
            prompt = random.choice(prompt_mixer_list).format(
                item=prompt, 
                style='<style-token>' if config['use_text_inversion'] else config['style_name'])

            print(prompt)
            all_images = generate_with_pipe(prompt, num_cols, num_rows, generator, config['num_inference_steps'])        
        
            os.mkdir(os.path.join(target_dir, str(i + 1), str(epoch)))
            for idx, image in enumerate(all_images):
                image.save(os.path.join(target_dir, str(i + 1), str(epoch), '{}.png'.format(idx)))

            grid = image_grid(all_images, num_rows, num_cols, config['output_size'])
            grid.save(os.path.join(target_dir, str(i + 1), 'grid_{}.png'.format(epoch)))
