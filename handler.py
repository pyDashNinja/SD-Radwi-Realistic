import runpod
import json
from GFPGAN import inference
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from transformers import pipeline, set_seed
import random
import re
from PIL import Image
import torch
from clip_interrogator import Config, Interrogator
import ast
import numpy as np
import cv2
from utils import ImageResize
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
import os
os.environ['HF_HOME'] = "./huggingface"

# """ --------------------------------------------- """

def image_to_prompt(image, MODELS , ci ,clip_model_name):

    ci.config.blip_num_beams = 64
    ci.config.chunk_size = 2048
    ci.config.flavor_intermediate_count = 2048 if clip_model_name == MODELS[0] else 1024

    image = image.convert('RGB')
    prompt = ci.interrogate_fast(image)

    return prompt

def generate(starting_text, gpt2_pipe):
    with open("name.txt", "r") as f:
        line = f.readlines()
    for count in range(1):
        seed = random.randint(100, 1000000)
        set_seed(seed)

        # If the text field is empty
        if starting_text == "":
            starting_text: str = line[random.randrange(0, len(line))].replace(
                "\n", "").lower().capitalize()
            starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)
            print(starting_text)

        response = gpt2_pipe(starting_text, max_length=random.randint(
            60, 90), num_return_sequences=1)
        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp)

        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+', '', response_end)
        response_end = response_end.replace("<", "").replace(">", "")
        if response_end != "":
            return response_end
        if count == 0:
            return response_end


# """ --------------------------------------------- """



def handler(event):
    try:
        data = event['input']
        operation = data['operation']
        if operation == 'text2image':
            model_id = "runwayml/stable-diffusion-v1-5"
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None, vae=vae)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
            # pipe.enable_model_cpu_offload()
            pipe.enable_xformers_memory_efficient_attention()

            pipe = pipe.to("cuda")
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = './weights/realesr-general-x4v3.pth'
            half = True if torch.cuda.is_available() else False
            upsampler = RealESRGANer(scale=4, model_path=model_path,
                             model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            print("Model loaded")
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
            try:
                lora = str(data['lora'])
                print("lora", lora)

                if lora:
                    lora_strength = float(data['lora_strength'])
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
                lora_strength = 1
                lora_filename = "add_detail.safetensors"
                pipe.load_lora_weights(
                        f"./lora/{lora_filename.split('.')[0]}", weight_name=lora_filename)
                lora_prompt = "<lora:add_detail:" + \
                    str(lora_strength) + ">, "

            try:
                height = int(ast.literal_eval(data['height']))
                width = int(ast.literal_eval(data['width']))
            except:
                height = 512
                width = 512
            # if len(prompt:
            prompt = lora_prompt + prompt + """, Ultra-high resolution, HDR, Vivid colors, Fine-grained textures, Enhanced details, Crisp and sharp image, Immersive depth of field, Professional-grade clarity, True-color representation"""
            image = pipe(prompt=prompt, negative_prompt=nprompt, width=width, height=height,
                         guidance_scale=gscale, num_inference_steps=steps, generator=generator).images[0]

            try:
                # image = image.convert("RGBA")
                scale = 4
                # convert image into numpy
                image = np.array(image)
                image = upsampler.enhance(image, outscale=scale)
                # print(image.shape, "image shape")
                # print("image", image)
                image = image[0]
                # print("image", image)
                image = Image.fromarray(image)
            except Exception as E:
                print(E)



            image = image.convert("RGB")
            # image.save("sd-generated-image.jpeg")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return json.dumps({"image": img_str}), 200

        elif operation == 'image2image':
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
            image = data['control_image']
            low = int(ast.literal_eval(data['low']))
            high = int(ast.literal_eval(data['high']))
            guessmode = bool(data['guessmode'])
            checkpoint = "lllyasviel/sd-controlnet-canny"

            controlnet = ControlNetModel.from_pretrained(
                checkpoint, torch_dtype=torch.float16)
            control = StableDiffusionControlNetPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V2.0", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
            )

            control.scheduler = UniPCMultistepScheduler.from_config(
                control.scheduler.config)
            # control.enable_model_cpu_offload()
            control.enable_xformers_memory_efficient_attention()
            control = control.to("cuda")

            try:
                lora = str(data['lora'])

                if lora:
                    lora_strength = float(data['lora_strength'])
                    lora_filename = f"{lora}.safetensors"
                    control.load_lora_weights(
                        f"./lora/{lora}", weight_name=lora_filename)

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
            try:
                # image = image.convert("RGBA")
                scale = 4
                # convert image into numpy
                image = np.array(image)
                image = upsampler.enhance(image, outscale=scale)
                # print(image.shape, "image shape")
                image = image[0]
                # print("image", image)
                image = Image.fromarray(image)
            except Exception as E:
                print(E)



            image = image.convert("RGB")
            # image.save("sd-generated-image.jpeg")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return json.dumps({"image": img_str}), 200

        elif operation == "faceenhance":
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = './weights/realesr-general-x4v3.pth'
            half = True if torch.cuda.is_available() else False
            upsampler = RealESRGANer(scale=4, model_path=model_path,
                                    model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

            face_enhancer = GFPGANer(
                model_path='./weights/GFPGANv1.3.pth', upscale=2, bg_upsampler=upsampler, arch='clean', channel_multiplier=2)
            image = data['image']
            # convert this base64string image into cv2 image
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
                # image resize
                image = image.resize((width, height))
            image = np.array(image)
            image = inference(face_enhancer, image)

            image = image.convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return json.dumps({"image": img_str}), 200

        elif operation == "image2prompt":
            image = data['image']
            MODELS = ['ViT-L (best for Stable Diffusion 1.*)']
            # image_to_prompt(image, MODELS , ci ,clip_model_name)
            # convert image from base64string to PIL image
            image = Image.open(BytesIO(base64.b64decode(image)))
            # load BLIP and ViT-L https://huggingface.co/openai/clip-vit-large-patch14
            config = Config(clip_model_name="ViT-L-14/openai")
            ci = Interrogator(config)

            prompt = image_to_prompt(
                image, MODELS, ci ,"ViT-L (best for Stable Diffusion 1.*)")
            print(prompt)
            return json.dumps({"generated": prompt}), 200
        
        elif operation == "promptgenerator":
            prompt = data['prompt']
            gpt2_pipe = pipeline(
                'text-generation', model='succinctly/text2image-prompt-generator')
            print("model loaded")
            while True:
                generated = generate(prompt, gpt2_pipe)
                if generated != "":
                    break
            print("generated", generated)
            return json.dumps({"generated": generated}), 200

        elif operation == "upscale":
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = './weights/realesr-general-x4v3.pth'
            half = True if torch.cuda.is_available() else False
            upsampler = RealESRGANer(scale=4, model_path=model_path,
                                    model=model, tile=0, tile_pad=10, pre_pad=0, half=half)
            image = data['image']
            # convert this base64string image into cv2 image
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
                # image resize
                image = image.resize((width, height))
            image = np.array(image)
            image = upsampler.enhance(image, outscale=4)
            image = image[0]
            image = Image.fromarray(image)
            image = image.convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return json.dumps({"image": img_str}), 200



    except Exception as E:
        print(E)
        return json.dumps({"message": "error something is wrong : " + str(E)}), 400



runpod.serverless.start({
    "handler": handler
})
