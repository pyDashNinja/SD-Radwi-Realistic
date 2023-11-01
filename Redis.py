import queue
import threading
import time
import redis
import os
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
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
    # StableDiffusionUpscalePipeline

)

os.environ['HF_HOME'] = "./huggingface"

request_queue = "request_queue"  # Name of the Redis queue
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

model_id = "stablediffusionapi/revanimated"
print("model loaded")
pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)
# pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

pipe = pipe.to("cuda")

checkpoint = "lllyasviel/sd-controlnet-canny"

controlnet = ControlNetModel.from_pretrained(
    checkpoint, torch_dtype=torch.float16)
control = StableDiffusionControlNetPipeline.from_pretrained(
    "stablediffusionapi/revanimated", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
)

control.scheduler = UniPCMultistepScheduler.from_config(
    control.scheduler.config)
# control.enable_model_cpu_offload()
control.enable_xformers_memory_efficient_attention()
control = control.to("cuda")


def process_request(data):
    try:
        gc.collect()
        data_str = data.decode('utf-8')
        data = ast.literal_eval(data_str)
        name = str(data['name'])
        # print(data)
        prompt = data['prompt']

        nprompt = data['nprompt']
        # print("----------------------------------------------")
        # print(prompt)
        seed = int(ast.literal_eval(data['seed']))
        steps = int(ast.literal_eval(data['steps']))
        gscale = float(ast.literal_eval(data['gscale']))
        model = str(data['model'])

        generator = torch.Generator()
        generator.manual_seed(seed)
        print("prompt 1", prompt)

        if model == 'base':
            try:
                lora = str(data['lora'])
                print("lora", lora)

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
                # lora_strength = 1
                # lora_filename = "add_detail.safetensors"
                # pipe.load_lora_weights(
                #         f"./lora/{lora_filename.split('.')[0]}", weight_name=lora_filename)
                # lora_prompt = "<lora:add_detail:" + \
                #     str(lora_strength) + ">, "
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
                # " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality | hard focus, smooth, depth of field, 8K UHD"
            # " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality, studio shoot, Nikon D850"
            # compel = Compel(tokenizer=pipe.tokenizer,
            #                 text_encoder=pipe.text_encoder)
            # prompt_embeds = compel.build_conditioning_tensor(
            #     prompt) if prompt else None
            print(prompt,"checkign prompt now")
            image = pipe(prompt=prompt, negative_prompt=nprompt, width=width, height=height,
                         guidance_scale=gscale, num_inference_steps=steps, generator=generator).images[0]

            pipe.unload_lora_weights()

            # low_threshold = 100
            # high_threshold = 200

            # # convert image into numpy
            # image = np.array(image)

            # image = cv2.Canny(image, low_threshold, high_threshold)
            # image = image[:, :, None]
            # image = np.concatenate([image, image, image], axis=2)
            # control_image = Image.fromarray(image)

            # print(prompt, "prompt")
            # # try:
            # #     height = int(ast.literal_eval(data['height']))
            # #     width = int(ast.literal_eval(data['width']))
            # # except:
            # height = 1024
            # width = 1024
            # # print(height, width, "height and width")

            # # image.save("sd-generated-image.jpeg")
            # # image.save('sd-generated-image.jpeg',
            # #            optimize=True, quality=80)
            # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
            #                 generator=generator, image=control_image).images[0]

        # read image as pillow and convert it to base64string image
        elif model == 'control':
            image = data['control_image']
            low = int(ast.literal_eval(data['low']))
            high = int(ast.literal_eval(data['high']))
            guessmode = bool(data['guessmode'])

            try:
                lora = str(data['lora'])

                if lora:
                    lora_strength = float(data['lora_strength'])
                    lora_filename = f"{lora}.safetensors"
                    control.load_lora_weights(
                        f"./lora/{lora}", weight_name=lora_filename)
                    lora_prompt = ""

                    if lora == 'add_detail':
                        lora_prompt = "<lora:add_detail:" + \
                            str(lora_strength) + ">, "
                            
                else:
                    lora_prompt = ""

            except:
                # lora_strength = 1
                # lora_filename = "add_detail.safetensors"
                # control.load_lora_weights(
                #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
                # lora_prompt = "<lora:add_detail:" + \
                #     str(lora_strength) + ">, "
                lora_prompt = ""
            print("prompt 1", prompt)
            print("lora prompt", lora_prompt)

            # convert image from base64string to PIL image
            image = Image.open(BytesIO(base64.b64decode(image)))
            # image.save("control.png")
            # print(image.size)
            # # get the image width and height
            # width, height = image.size
            # if width*height > 3000*3000:
            #     width = int(width*0.2)
            #     height = int(height*0.2)
            # elif width*height > 2000*2000:
            #     width = int(width*0.4)
            #     height = int(height*0.4)
            # elif width*height > 1000*1000:
            #     width = int(width*0.6)
            #     height = int(height*0.6)
            # elif width*height > 500*500:
            #     width = int(width*0.8)
            #     height = int(height*0.8)
            # print(width, height, "new width and height")
            # # check if height and width is divisible by 8
            # if width % 8 != 0:
            #     width = width - (width % 8)
            # if height % 8 != 0:
            #     height = height - (height % 8)

            # image = image.resize((width, height))

            # image = depth_estimator(image)['predicted_depth'][0]
            # image = image.numpy()

            # image_depth = image.copy()
            # image_depth -= np.min(image_depth)
            # image_depth /= np.max(image_depth)

            # bg_threhold = 0.4

            # x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            # x[image_depth < bg_threhold] = 0

            # y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            # y[image_depth < bg_threhold] = 0

            # z = np.ones_like(x) * np.pi * 2.0

            # image = np.stack([x, y, z], axis=2)
            # image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
            # image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            # image = Image.fromarray(image)
            image = np.array(image)

            low_threshold = low
            high_threshold = high

            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)
            control_image.save('sd-generated-image.jpeg',
                               optimize=True, quality=80)
            # change size of image
            prompt = lora_prompt + prompt
            print(prompt, "prompt")
            try:
                height = int(ast.literal_eval(data['height']))
                width = int(ast.literal_eval(data['width']))
            except:
                height = 512
                width = 512
            print(height, width, "height and width")

            # image.save("sd-generated-image.jpeg")
            # image.save('sd-generated-image.jpeg',
            #            optimize=True, quality=80)
            # compel = Compel(tokenizer=pipe.tokenizer,
            #                 text_encoder=pipe.text_encoder)
            # prompt_embeds = compel.build_conditioning_tensor(
            #     prompt) if prompt else None
            image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
                            generator=generator, guess_mode=guessmode, image=control_image).images[0]
            # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
            #                 generator=generator, image=image).images[0]
            control.unload_lora_weights()
            

        # elif model == 'upscale':
        #     print(model, "model")
        #     low_res_img = data['low_res_image']
        #     try:
        #         target_size = int(data['target_size'])
        #         constant = int(data['constant'])
        #         low_res_img = Image.open(
        #             BytesIO(base64.b64decode(low_res_img)))
        #         low_res_img = ImageResize(low_res_img, target_size, constant)
        #     except:
        #         low_res_img = Image.open(
        #             BytesIO(base64.b64decode(low_res_img)))

        #     # print( low res image size )
        #     print(low_res_img.size, "image size ---")

        #     try:
        #         lora = str(data['lora'])

        #         if lora == "add_detail":
        #             lora_strength = float(data['lora_strength'])
        #             lora_filename = "add_detail.safetensors"
        #             control.load_lora_weights(
        #                 "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)

        #             lora_prompt = "<lora:add_detail:" + \
        #                 str(lora_strength) + ">, "
        #         else:
        #             lora_prompt = ""

        #     except:
        #         # lora_strength = 1
        #         # lora_filename = "add_detail.safetensors"
        #         # control.load_lora_weights(
        #         #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
        #         # lora_prompt = "<lora:add_detail:" + \
        #         #     str(lora_strength) + ">, "
        #         lora_prompt = ""

        #     print("prompt 1", prompt)
        #     print("lora prompt", lora_prompt)
        #     prompt = lora_prompt + prompt
        #     image = upscale(prompt=prompt, negative_prompt=nprompt, guidance_scale=gscale,
        #                     num_inference_steps=steps, generator=generator, image=low_res_img).images[0]

        # image = Image.open('sd-generated-image.png')
        # convert image into webp
        image = image.convert("RGB")
        image.save(name)
    except Exception as E:
        # write error to a csv file with the name of the image, timestamp, error using pandas
        try:
            df = pd.read_csv('error.csv')
        except:
            df = pd.DataFrame(columns=['name', 'timestamp', 'error'])

        # append row to the dataframe
        df = df.append({'name': name, 'timestamp': time.time(), 'error': E}, ignore_index=True)

        # overwrite the csv file
        df.to_csv('error.csv', index=False)

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