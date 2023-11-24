import queue
import threading
import time
import redis
import os
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from transformers import pipeline, set_seed
import random
import re
import torch
from PIL import Image
import pandas as pd
import gc
import ast
import numpy as np
import cv2
from utils import ImageResize
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionInpaintPipelineLegacy
)
from transformers import pipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from ip_adapter import IPAdapter, IPAdapterPlus

os.environ['HF_HOME'] = "./huggingface"

request_queue = "request_queue"  # Name of the Redis queue
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

MODEL_NAME = "SG161222/Realistic_Vision_V5.1_noVAE"
MODEL_NAME_INPAINT = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_NAME = "stabilityai/sd-vae-ft-mse-original"
VAE_CKPT = "vae-ft-mse-840000-ema-pruned.ckpt"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"

vae = AutoencoderKL.from_single_file(
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
    torch_dtype=torch.float16
        )
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")


pipe_two = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    MODEL_NAME_INPAINT,
    torch_dtype=torch.float16,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

pipe_two.scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe_two.enable_xformers_memory_efficient_attention()

pipe_two = pipe_two.to("cuda")

image_encoder_path = "./models/image_encoder/"
ip_ckpt = "./models/ip-adapter-plus-face_sd15.bin"
device = "cuda"
ip_model = IPAdapterPlus(pipe_two, image_encoder_path, ip_ckpt, device, num_tokens=16)

def process_request(data):
    try:
        gc.collect()
        data_str = data.decode('utf-8')
        data = ast.literal_eval(data_str)
        model = str(data['model'])

        if model == 'base':
            name = str(data['name'])
            # print(data)
            prompt = data['prompt']
    
            nprompt = data['nprompt']
            # print("----------------------------------------------")
            # print(prompt)
            seed = int(ast.literal_eval(data['seed']))
            steps = int(ast.literal_eval(data['steps']))
            gscale = float(ast.literal_eval(data['gscale']))
            
            generator = torch.Generator()
            generator.manual_seed(seed)
            print("prompt 1", prompt)
            try:
                lora = str(data['lora'])
                print("lora", lora)#

                if lora != "None":
                    print("here inside lora")
                    lora_strength = float(data['lora_strength'])
                    print("here inside here here")
                    lora_filename = f"{lora}.safetensors"
                    print("lora filename", lora_filename)
                    pipe.load_lora_weights(
                        f"./lora/{lora}", weight_name=lora_filename)
                    print("model loaded")
                    lora_prompt = ""

                    if lora == 'add_detail':
                        print("add detail")
                        lora_prompt = "<lora:add_detail:" + \
                            str(lora_strength) + ">, "

                elif lora == "None":
                    print("lora none")
                    lora_prompt = ""

                else:
                    print("lora else")
                    lora_prompt = ""

            except Exception as E:
                print("Exception", E)
                lora_prompt = ""

            try:
                height = int(ast.literal_eval(data['height']))
                width = int(ast.literal_eval(data['width']))
            except:
                height = 512
                width = 512
            print(height, width, "height and width")
            # if len(prompt:
            prompt = lora_prompt + prompt
            image = pipe(prompt=prompt, negative_prompt=nprompt, width=width, height=height,
                         guidance_scale=gscale, num_inference_steps=steps, generator=generator).images[0]

            pipe.unload_lora_weights()
            
        elif model == 'control':
            image = data['control_image']
            name = str(data['name'])
           
            image = Image.open(BytesIO(base64.b64decode(image)))
            width, height = image.size
            if width*height >= 1024*1024:
              image = image.convert("RGB")
              # get size of image
              width, height = image.size
              # resize image in such a way width and height product never exceeds 1500*1500
              while width*height >= 1024*1024:
                width = int(width*0.8)
                height = int(height*0.8)
        # make width height divisible by 8
                width = width - (width % 8)
                height = height - (height % 8)
              image = image.resize((width, height))
              print(image.size,"image size")

            # find the image name
            template_masked_image = f'./template/{str(data['template'])}/masked_image.png'
            template_mask = f'./template/{str(data['template'])}/mask.png'
            masked_image = Image.open(template_masked_image).convert("RGB")
            mask = Image.open(template_mask).convert('RGB')

            try:
                seed = int(ast.literal_eval(data['seed']))
                steps = int(ast.literal_eval(data['steps']))
                strength = float(ast.literal_eval(data['strength']))
            except:
                seed = int(random.randint(0, 100000))
                steps = 30
                strength = 0.7

            
            image = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=steps, seed=seed,
                image=masked_image,mask_image=mask , strength=strength)

            try:
                image = image[0]
            except:
                pass
            print(image,"image")

        
        image = image.convert("RGB")
        image.save(name)
    except Exception as E:
        print(E,"exception")

def worker():
    while True:
        data = redis_client.lpop(request_queue)
        if data is None:
            pass
        else:
            process_request(data)
        time.sleep(1)

# Use a single-item processing queue to ensure only one request is processed at a time
processing_queue = queue.Queue()

# Start worker threads
num_workers = 1  # Adjust the number of worker threads as needed
workers = []
for _ in range(num_workers):
    t = threading.Thread(target=worker)
    t.start()
    workers.append(t)
