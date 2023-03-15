import os
import json
import shutil
import argparse
import itertools
from PIL import Image
from tqdm.auto import tqdm
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.utils.data import Dataset

from accelerate import Accelerator
from transformers import CLIPProcessor
from libs.augmentation import PicsartData
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', type=str, help='The path to the configuration file')
    args = parser.parse_args()
    
    config = json.load(open(args.config, 'r'))

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, 
        beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        config['pretrained_name'],
        subfolder="tokenizer",
    )

    if config['use_text_inversion']:
        initializer_token = config['style_name']
        placeholder_token = "<style-token>"        
        
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        assert num_added_tokens == 1

        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        assert len(token_ids) == 1

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    dataset = PicsartData(
        data_root=config['data_root'],
        source_list=config['source_list'],
        tokenizer=tokenizer,
        style_name=placeholder_token if config['use_text_inversion'] else config['style_name'],
        use_text_inversion=config['use_text_inversion'],
        augmentation=config['augmentation'],
        image_size=config['image_size'],
        clip_similars=config['clip_similars'] if 'clip_similars' in config.keys() else None
    )

    if 'resume' in config.keys():
        start_epoch = config['resume']['epoch']
        pipeline = StableDiffusionPipeline.from_pretrained(os.path.join(config['resume']['model_path']))
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        unet = pipeline.unet
        tokenizer = pipeline.tokenizer
    else:
        start_epoch = 0

        text_encoder = CLIPTextModel.from_pretrained(config['pretrained_name'], subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(config['pretrained_name'], subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(config['pretrained_name'], subfolder="unet")
        
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    if config['use_content_loss']:
        image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        image_encoder.requires_grad_(False)

    if config['freeze_vae']:
        vae.requires_grad_(False)    
    if config['freeze_unet']:
        unet.requires_grad_(False)
    if not config['use_text_inversion']:
        text_encoder.requires_grad_(False)
    else:
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

    num_train_epochs = config['train_epochs']
    timestep_range = config['training_timestep_range']
    output_dir = config['output_dir']

    accelerator = Accelerator()

    if accelerator.is_main_process and start_epoch == 0:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print('Saving checkpoints and demo images to:', output_dir)    

        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),        
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(os.path.join(output_dir, 'epoch0'))    

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = torch.optim.AdamW(
        (list(text_encoder.get_input_embeddings().parameters()) if config['use_text_inversion'] else []) + \
        (list(vae.parameters()) if not config['freeze_vae'] else []) + \
        (list(unet.parameters()) if not config['freeze_unet'] else []),
        lr=config['learning_rate'],
    )
    if len(config['step_milestones']) > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['step_milestones'], gamma=0.5)

    if config['use_text_inversion']:
        text_encoder = accelerator.prepare(text_encoder)
    if not config['freeze_vae']:
        vae = accelerator.prepare(vae)
    if not config['freeze_unet']:
        unet = accelerator.prepare(unet)
    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    vae.to(accelerator.device)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)
    if config['use_content_loss']:
        image_encoder.to(accelerator.device)
        image_encoder.eval()

    if config['use_text_inversion']:
        text_encoder.train()
    else:
        text_encoder.eval()
    if config['freeze_vae']:
        vae.eval()
    else:
        vae.train()
    if config['freeze_unet']:
        unet.eval()
    else:
        unet.train()

    progress_bar = tqdm(range(num_train_epochs), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Epochs")

    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for epoch in tqdm(range(num_train_epochs)):
        if epoch >= start_epoch:
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                if config['use_sparse_update']:
                    timesteps = torch.randint(timestep_range[0], timestep_range[1], (bsz,), device=latents.device).long()
                    timesteps = timesteps * (1000 // config['num_inference_steps'])
                else:
                    timesteps = torch.randint(timestep_range[0], timestep_range[1], (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)        

                # Predict the noise residual and calculate denoise loss
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                ldm_loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                if config['use_content_loss']:
                    caption_hidden_states = text_encoder(batch["raw_ids"])[0]
                    # Generate another image with same propmt and calculate content loss
                    eval_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
                                                   beta_schedule="scaled_linear", skip_prk_steps=True)

                    eval_scheduler.set_timesteps(config['num_inference_steps'])
                    if torch.is_tensor(eval_scheduler.timesteps):
                        timesteps_tensor = eval_scheduler.timesteps
                    else:
                        timesteps_tensor = torch.tensor(eval_scheduler.timesteps.copy())

                    uncond_input = tokenizer([""] * latents.shape[0], padding="max_length",
                                             max_length=tokenizer.model_max_length, return_tensors="pt")
                    uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]
                    text_embeddings = torch.cat([uncond_embeddings, caption_hidden_states])
                                    
                    fake_latents = torch.randn(
                        latents.shape,
                        device=latents.device,
                        dtype=latents.dtype,
                    )              
                        
                    for t in timesteps_tensor:                                           
                        if t == timesteps[0]:
                            latent_model_input = torch.cat([fake_latents] * 2)
                            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                            fake_latents = eval_scheduler.step(noise_pred, t, fake_latents).prev_sample
                        else:
                            with torch.no_grad():
                                latent_model_input = torch.cat([fake_latents] * 2)
                                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                                fake_latents = eval_scheduler.step(noise_pred, t, fake_latents).prev_sample

                    fake_latents = 1 / 0.18215 * fake_latents
                    fake_image = vae.decode(fake_latents).sample
                    fake_image = (fake_image / 2 + 0.5).clamp(0, 1)
                    fake_image = ((fake_image.detach().cpu().permute(0, 2, 3, 1).numpy()) * 255).round().astype("uint8")        
                    fake_image = Image.fromarray(fake_image[0])
                    
                    fake_input = image_processor(images=fake_image, return_tensors='pt')['pixel_values'].to(accelerator.device)
                    fake_feature = image_encoder(pixel_values=fake_input).pooler_output
                    gt_input = image_processor(images=Image.open(batch['image_path'][0]), return_tensors='pt')['pixel_values'].to(accelerator.device)
                    gt_feature = image_encoder(pixel_values=gt_input).pooler_output
                    
                    content_loss = 1 - cosine_similarity(fake_feature, gt_feature)                                        
                else:
                    content_loss = 0
                
                loss = ldm_loss + content_loss * config['content_weight']                        

                logs = {
                    "ldm_loss": ldm_loss.detach().item(),
                    "content_loss": 0 if not config['use_content_loss'] else content_loss.detach().item(),
                    "loss": loss.detach().item(),
                }
                progress_bar.set_postfix(**logs)       
                
                accelerator.backward(loss)
                
                if config['use_text_inversion']:
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()

        progress_bar.update(1)
        accelerator.wait_for_everyone()
        if len(config['step_milestones']) > 0:
            scheduler.step()

        if epoch >= start_epoch and accelerator.is_main_process and epoch + 1 in config['checkpoints']:
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                unet=accelerator.unwrap_model(unet),
                tokenizer=tokenizer,
                scheduler=PNDMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                ),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            pipeline.save_pretrained(os.path.join(output_dir, 'epoch{}'.format(epoch + 1)))

    print('Finished!')
