from GFPGAN import inference
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from transformers import pipeline, set_seed
import random
import re
import torch
from PIL import Image
import pandas as pd
import torch
import gc
from clip_interrogator import Config, Interrogator
import ast
import numpy as np
import cv2
from utils import ImageResize
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
# from compel import Compel
# from diffusers.models import 
import os
from flask_cors import CORS
from functools import wraps
import redis
import time
import io
import base64
from PIL import Image


os.environ['HF_HOME'] = "./huggingface"



from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    # StableDiffusionUpscalePipeline

)
def before_first_request(f):
    already_run = False
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        nonlocal already_run
        if not already_run:
            already_run = True
            f(*args, **kwargs)
            
    
    return wrapper

# import redis
# from rq import Queue

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
request_queue = "request_queue"  # Name of the Redis queue

app = Flask(__name__)
CORS(app)
# allow all origins
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}


# r = redis.Redis()
# q = Queue(connection=r)


with open("name.txt", "r") as f:
    line = f.readlines()


def image_to_prompt(image, clip_model_name):

    ci.config.blip_num_beams = 64
    ci.config.chunk_size = 2048
    ci.config.flavor_intermediate_count = 2048 if clip_model_name == MODELS[0] else 1024

    image = image.convert('RGB')
    prompt = ci.interrogate_fast(image)

    return prompt


def generate(starting_text):
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


@app.before_request
@before_first_request
def load_model():
    # check if the variable is defined in globals
    # print("Loading model")

    # global pipe
    # # model_id = "prompthero/openjourney"
    # # model_id = "stabilityai/stable-diffusion-2"
    # # model_id = "Lykon/DreamShaper"
    # model_id = "stablediffusionapi/revanimated"
    

    # # model_id= "stablediffusionapi/anything-v5"
    # # pipe = StableDiffusionXLPipeline.from_pretrained(
    # #     "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    # # )
    # # pipe.enable_model_cpu_offload()
    # # pipe.unet = torch.compile(
    # #     pipe.unet, mode="reduce-overhead", fullgraph=True)
    # # pipe.enable_xformers_memory_efficient_attention()

    # # scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    # #     model_id, subfolder="scheduler")
    # # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model_id, torch_dtype=torch.float16, safety_checker=None)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #     pipe.scheduler.config, use_karras_sigmas=True
    # )
    # # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    # pipe = pipe.to("cuda")
    # lora_model_path = "OedoSoldier/detail-tweaker-lora"

    # pipe.load_lora_weights(".", weight_name=lora_filename)
    #

    print("Model loaded")

    # check if models/GFPGANv1.3.pth exists
    global face_enhancer
    global upsampler
    import os
    if not os.path.exists("./weights/GFPGANv1.4.pth") or not os.path.exists("./weights/realesr-general-x4v3.pth"):
        os.system(
            "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
        os.system(
            "wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")

        # move the file GFPGANv1.3.pth to the weights folder
        os.system("mv GFPGANv1.4.pth ./weights")
        os.system("mv realesr-general-x4v3.pth ./weights")

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    model_path = './weights/realesr-general-x4v3.pth'
    half = True if torch.cuda.is_available() else False
    upsampler = RealESRGANer(scale=4, model_path=model_path,
                             model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

    face_enhancer = GFPGANer(
        model_path='./weights/GFPGANv1.4.pth', upscale=2, bg_upsampler=upsampler, arch='clean', channel_multiplier=2)

    # check if pipe is in cuda
    # print(pipe.device)
    # print("Model in cuda: ", next(pipe.parameters()).is_cuda)
    # global control

    # checkpoint = "lllyasviel/sd-controlnet-canny"

    # controlnet = ControlNetModel.from_pretrained(
    #     checkpoint, torch_dtype=torch.float16)
    # control = StableDiffusionControlNetPipeline.from_pretrained(
    #     "Lykon/DreamShaper", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    # )

    # control.scheduler = UniPCMultistepScheduler.from_config(
    #     control.scheduler.config)
    # # control.enable_model_cpu_offload()
    # control.enable_xformers_memory_efficient_attention()
    # control = control.to("cuda")

    # global upscale

    # model_id = "stabilityai/stable-diffusion-x4-upscaler"
    # upscale = StableDiffusionUpscalePipeline.from_pretrained(
    #     model_id, torch_dtype=torch.float16)
    # upscale.enable_xformers_memory_efficient_attention()
    # upscale = upscale.to("cuda")

    # global control
    # global depth_estimator

    # depth_estimator = pipeline(
    #     "depth-estimation", model="Intel/dpt-hybrid-midas")

    # controlnet = ControlNetModel.from_pretrained(
    #     "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
    # )

    # control = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    # )

    # control.scheduler = UniPCMultistepScheduler.from_config(
    #     control.scheduler.config)

    # # Remove if you do not have xformers installed
    # # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # # for installation instructions
    # control.enable_xformers_memory_efficient_attention()

    # control.enable_model_cpu_offload()

    global gpt2_pipe
    gpt2_pipe = pipeline(
        'text-generation', model='succinctly/text2image-prompt-generator')
    print("model loaded")
    # , 'ViT-H (best for Stable Diffusion 2.*)']
    global MODELS
    global ci
    MODELS = ['ViT-L (best for Stable Diffusion 1.*)']

    # load BLIP and ViT-L https://huggingface.co/openai/clip-vit-large-patch14
    config = Config(clip_model_name="ViT-L-14/openai")
    ci = Interrogator(config)


@app.route("/imageprompt", methods=["POST"])
def imageprompt():
    try:
        gc.collect()
        data = request.json
        # print(data)
        image = data['image']
        # convert image from base64string to PIL image
        image = Image.open(BytesIO(base64.b64decode(image)))

        prompt = image_to_prompt(
            image, "ViT-L (best for Stable Diffusion 1.*)")
        print(prompt)
        return jsonify({"generated": prompt}), 200
    except Exception as E:
        print(E)
        return jsonify({"message": "error something is wrong : " + str(E)}), 400


@app.route('/promptgenerate', methods=['POST'])
def promptgenerate():
    # get args
    try:
        gc.collect()
        data = request.json
        # print(data)
        prompt = data['prompt']
        while True:
            generated = generate(prompt)
            if generated != "":
                break
        print("generated", generated)
        return jsonify({"generated": generated}), 200
    except Exception as E:
        print(E)
        return jsonify({"message": "error something is wrong : " + str(E)}), 400


@app.route('/faceenhance', methods=['POST'])
def faceenhance():
    try:
        data = request.json
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
        return jsonify({"image": img_str}), 200
    except:
        return jsonify({"message": "error something is wrong"}), 400


@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        scale = 4
        data = request.json
        image = data['image']
        # convert this base64string image into cv2 image
        image = Image.open(BytesIO(base64.b64decode(image)))
        # get the size of the image
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
        image = upsampler.enhance(image, outscale=scale)
        image = image[0]
        # print("image", image)
        image = Image.fromarray(image)
        image = image.convert("RGB")
        # image.save("sd-generated-image.jpeg")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return jsonify({"image": img_str}), 200
    except Exception as E:
        print(E, "Error")
        return jsonify({"message": "error something is wrong"}), 400


@app.route('/', methods=['POST', 'GET'])
def initv2():
    try:
        # check if get request 
        if request.method == 'GET':
            return "http://18.222.34.232:5000"

        data = request.json
        print(data, "data")
        # add "name" variable in data object which is data = {"prompt" : "hello"}
        name = str(int(str(time.time()).split(".")[0]) + random.randint(100, 1000000)) + ".jpeg"
        data['name'] = name

        redisthings = redis_client.rpush(request_queue, str(data))

        # start time
        start = time.time()
        while True:
            time.sleep(0.5)
            # check if f"{name}.jpg" file exists
            if os.path.exists(name):
                time.sleep(0.5)
                # get the currect directory
                cwd = os.getcwd()
                # join path
                name = os.path.join(cwd, name)
                
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                try:
                    # image = image.convert("RGBA")
                    scale = 4
                    # convert image into numpy
                    image = np.array(image)
                    image = upsampler.enhance(image, outscale=scale)
                    # print(image.shape, "image shape")
                    # print("image", image)
                except Exception as E:
                    print(E)
                
                print("done scale out")

                image = image[0]
                # print("image", image)
                image = Image.fromarray(image)
                # compress image

                compressed_image = io.BytesIO()
                image.save(compressed_image, format='JPEG', quality=50, optimize=True)
                compressed_image.seek(0)  # Reset the stream position

                # Convert the compressed image to base64 string
                img_str = base64.b64encode(compressed_image.read()).decode('utf-8')
                try:
                    # return the content
                    return jsonify({"image": img_str}), 200
                except Exception as E:
                    pass
                finally:
                    try:
                        special = bool(data['special'])
                    except:
                        special = False
                    
                    if not special:
                        os.remove(name)

            # get end time
            end = time.time()
            # now check if it has been 3 minutes then break
            if end - start > 180:
                break
            

        return jsonify({"message": "unsuccessfull"}), 200
    except Exception as E:
        print(E)
        return jsonify({"message": "Error: " + str(E)}), 400


# @app.route('/v2', methods=['POST', 'GET'])
# def init():
#     # try:
#     #     job = q.enqueue(background_process, request)
#     #     # return jsonify({"message": "success"}), 200
#     # except Exception as E:

    

#     try:
#         # check method if it is GET
#         if request.method == 'GET':
#             return "http://18.222.34.232:5000"
#     except:
#         pass
#     try:
#         gc.collect()
#         # get the json file
#         # print(request.json)
#         data = request.json
#         # print(data)
#         prompt = data['prompt']

#         nprompt = data['nprompt']
#         # print("----------------------------------------------")
#         # print(prompt)
#         seed = int(ast.literal_eval(data['seed']))
#         steps = int(ast.literal_eval(data['steps']))
#         gscale = float(ast.literal_eval(data['gscale']))
#         model = str(data['model'])

#         generator = torch.Generator()
#         generator.manual_seed(seed)
#         print("prompt 1", prompt)

#         if model == 'base':
#             try:
#                 lora = str(data['lora'])
#                 print("lora", lora)

#                 if lora:
#                     lora_strength = float(data['lora_strength'])
#                     lora_filename = f"{lora}.safetensors"
#                     print("lora filename", lora_filename)
#                     pipe.load_lora_weights(
#                         f"./lora/{lora}", weight_name=lora_filename)
#                     print("model loaded")
#                     lora_prompt = ""

#                     if lora == 'add_detail':
#                         print("add detail")
#                         lora_prompt = "<lora:add_detail:" + \
#                             str(lora_strength) + ">, "

#                 elif lora == "None":
#                     print("lora none")
#                     lora_prompt = ""

#                 else:
#                     print("lora else")
#                     lora_prompt = ""

#             except Exception as E:
#                 # print("Exception", E)
#                 # lora_strength = 1
#                 # lora_filename = "add_detail.safetensors"
#                 # pipe.load_lora_weights(
#                 #         f"./lora/{lora_filename.split('.')[0]}", weight_name=lora_filename)
#                 # lora_prompt = "<lora:add_detail:" + \
#                 #     str(lora_strength) + ">, "
#                 lora_prompt = ""

#             try:
#                 height = int(ast.literal_eval(data['height']))
#                 width = int(ast.literal_eval(data['width']))
#             except:
#                 height = 512
#                 width = 512
#             print(height, width, "height and width")
#             # if len(prompt:
#             prompt = lora_prompt + prompt
#                 # " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality | hard focus, smooth, depth of field, 8K UHD"
#             # " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality, studio shoot, Nikon D850"
#             # compel = Compel(tokenizer=pipe.tokenizer,
#             #                 text_encoder=pipe.text_encoder)
#             # prompt_embeds = compel.build_conditioning_tensor(
#             #     prompt) if prompt else None
#             print(prompt,"checkign prompt now")
#             image = pipe(prompt=prompt, negative_prompt=nprompt, width=width, height=height,
#                          guidance_scale=gscale, num_inference_steps=steps, generator=generator).images[0]

#             pipe.unload_lora_weights()

#             try:
#                 # image = image.convert("RGBA")
#                 scale = 4
#                 # convert image into numpy
#                 image = np.array(image)
#                 image = upsampler.enhance(image, outscale=scale)
#                 # print(image.shape, "image shape")
#                 # print("image", image)
#             except Exception as E:
#                 print(E)

#             # low_threshold = 100
#             # high_threshold = 200

#             # # convert image into numpy
#             # image = np.array(image)

#             # image = cv2.Canny(image, low_threshold, high_threshold)
#             # image = image[:, :, None]
#             # image = np.concatenate([image, image, image], axis=2)
#             # control_image = Image.fromarray(image)

#             # print(prompt, "prompt")
#             # # try:
#             # #     height = int(ast.literal_eval(data['height']))
#             # #     width = int(ast.literal_eval(data['width']))
#             # # except:
#             # height = 1024
#             # width = 1024
#             # # print(height, width, "height and width")

#             # # image.save("sd-generated-image.jpeg")
#             # # image.save('sd-generated-image.jpeg',
#             # #            optimize=True, quality=80)
#             # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#             #                 generator=generator, image=control_image).images[0]

#         # read image as pillow and convert it to base64string image
#         elif model == 'control':
#             image = data['control_image']
#             low = int(ast.literal_eval(data['low']))
#             high = int(ast.literal_eval(data['high']))
#             guessmode = bool(data['guessmode'])

#             try:
#                 lora = str(data['lora'])

#                 if lora:
#                     lora_strength = float(data['lora_strength'])
#                     lora_filename = f"{lora}.safetensors"
#                     control.load_lora_weights(
#                         f"./lora/{lora}", weight_name=lora_filename)

#                     if lora == 'add_detail':
#                         lora_prompt = "<lora:add_detail:" + \
#                             str(lora_strength) + ">, "
#                 else:
#                     lora_prompt = ""

#             except:
#                 # lora_strength = 1
#                 # lora_filename = "add_detail.safetensors"
#                 # control.load_lora_weights(
#                 #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
#                 # lora_prompt = "<lora:add_detail:" + \
#                 #     str(lora_strength) + ">, "
#                 lora_prompt = ""
#             print("prompt 1", prompt)
#             print("lora prompt", lora_prompt)

#             # convert image from base64string to PIL image
#             image = Image.open(BytesIO(base64.b64decode(image)))
#             # image.save("control.png")
#             # print(image.size)
#             # # get the image width and height
#             # width, height = image.size
#             # if width*height > 3000*3000:
#             #     width = int(width*0.2)
#             #     height = int(height*0.2)
#             # elif width*height > 2000*2000:
#             #     width = int(width*0.4)
#             #     height = int(height*0.4)
#             # elif width*height > 1000*1000:
#             #     width = int(width*0.6)
#             #     height = int(height*0.6)
#             # elif width*height > 500*500:
#             #     width = int(width*0.8)
#             #     height = int(height*0.8)
#             # print(width, height, "new width and height")
#             # # check if height and width is divisible by 8
#             # if width % 8 != 0:
#             #     width = width - (width % 8)
#             # if height % 8 != 0:
#             #     height = height - (height % 8)

#             # image = image.resize((width, height))

#             # image = depth_estimator(image)['predicted_depth'][0]
#             # image = image.numpy()

#             # image_depth = image.copy()
#             # image_depth -= np.min(image_depth)
#             # image_depth /= np.max(image_depth)

#             # bg_threhold = 0.4

#             # x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#             # x[image_depth < bg_threhold] = 0

#             # y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
#             # y[image_depth < bg_threhold] = 0

#             # z = np.ones_like(x) * np.pi * 2.0

#             # image = np.stack([x, y, z], axis=2)
#             # image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
#             # image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
#             # image = Image.fromarray(image)
#             image = np.array(image)

#             low_threshold = low
#             high_threshold = high

#             image = cv2.Canny(image, low_threshold, high_threshold)
#             image = image[:, :, None]
#             image = np.concatenate([image, image, image], axis=2)
#             control_image = Image.fromarray(image)
#             control_image.save('sd-generated-image.jpeg',
#                                optimize=True, quality=80)
#             # change size of image
#             prompt = lora_prompt + prompt
#             print(prompt, "prompt")
#             try:
#                 height = int(ast.literal_eval(data['height']))
#                 width = int(ast.literal_eval(data['width']))
#             except:
#                 height = 512
#                 width = 512
#             print(height, width, "height and width")

#             # image.save("sd-generated-image.jpeg")
#             # image.save('sd-generated-image.jpeg',
#             #            optimize=True, quality=80)
#             # compel = Compel(tokenizer=pipe.tokenizer,
#             #                 text_encoder=pipe.text_encoder)
#             # prompt_embeds = compel.build_conditioning_tensor(
#             #     prompt) if prompt else None
#             image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#                             generator=generator, guess_mode=guessmode, image=control_image).images[0]
#             # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#             #                 generator=generator, image=image).images[0]
#             control.unload_lora_weights()
#             try:
#                 # image = image.convert("RGBA")
#                 scale = 4
#                 # convert image into numpy
#                 image = np.array(image)
#                 image = upsampler.enhance(image, outscale=scale)
#                 # print(image.shape, "image shape")
#                 # print("image", image)
#             except Exception as E:
#                 print(E)

#         # elif model == 'upscale':
#         #     print(model, "model")
#         #     low_res_img = data['low_res_image']
#         #     try:
#         #         target_size = int(data['target_size'])
#         #         constant = int(data['constant'])
#         #         low_res_img = Image.open(
#         #             BytesIO(base64.b64decode(low_res_img)))
#         #         low_res_img = ImageResize(low_res_img, target_size, constant)
#         #     except:
#         #         low_res_img = Image.open(
#         #             BytesIO(base64.b64decode(low_res_img)))

#         #     # print( low res image size )
#         #     print(low_res_img.size, "image size ---")

#         #     try:
#         #         lora = str(data['lora'])

#         #         if lora == "add_detail":
#         #             lora_strength = float(data['lora_strength'])
#         #             lora_filename = "add_detail.safetensors"
#         #             control.load_lora_weights(
#         #                 "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)

#         #             lora_prompt = "<lora:add_detail:" + \
#         #                 str(lora_strength) + ">, "
#         #         else:
#         #             lora_prompt = ""

#         #     except:
#         #         # lora_strength = 1
#         #         # lora_filename = "add_detail.safetensors"
#         #         # control.load_lora_weights(
#         #         #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
#         #         # lora_prompt = "<lora:add_detail:" + \
#         #         #     str(lora_strength) + ">, "
#         #         lora_prompt = ""

#         #     print("prompt 1", prompt)
#         #     print("lora prompt", lora_prompt)
#         #     prompt = lora_prompt + prompt
#         #     image = upscale(prompt=prompt, negative_prompt=nprompt, guidance_scale=gscale,
#         #                     num_inference_steps=steps, generator=generator, image=low_res_img).images[0]

#         # image = Image.open('sd-generated-image.png')
#         # convert image into webp
#         image = image[0]
#         # print("image", image)
#         image = Image.fromarray(image)
#         image = image.convert("RGB")
#         # image.save("sd-generated-image.jpeg")

#         buffered = BytesIO()
#         image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         return jsonify({"image": img_str}), 200
#     except Exception as E:
#         print(E)
#         return jsonify({"message": "error something is wrong : " + str(E)}), 400


# def background_process(request):
#     try:
#         # check method if it is GET
#         if request.method == 'GET':
#             return "http://18.222.34.232:5000"
#     except:
#         pass
#     try:
#         gc.collect()
#         # get the json file
#         # print(request.json)
#         data = request.json
#         # print(data)
#         prompt = data['prompt']

#         nprompt = data['nprompt']
#         # print("----------------------------------------------")
#         # print(prompt)
#         seed = int(ast.literal_eval(data['seed']))
#         steps = int(ast.literal_eval(data['steps']))
#         gscale = float(ast.literal_eval(data['gscale']))
#         model = str(data['model'])

#         generator = torch.Generator()
#         generator.manual_seed(seed)
#         print("prompt 1", prompt)

#         if model == 'base':
#             try:
#                 lora = str(data['lora'])
#                 print("lora", lora)

#                 if lora:
#                     lora_strength = float(data['lora_strength'])
#                     lora_filename = f"{lora}.safetensors"
#                     print("lora filename", lora_filename)
#                     pipe.load_lora_weights(
#                         f"./lora/{lora}", weight_name=lora_filename)
#                     print("model loaded")
#                     lora_prompt = ""

#                     if lora == 'add_detail':
#                         print("add detail")
#                         lora_prompt = "<lora:add_detail:" + \
#                             str(lora_strength) + ">, "

#                 elif lora == "None":
#                     print("lora none")
#                     lora_prompt = ""

#                 else:
#                     print("lora else")
#                     lora_prompt = ""

#             except Exception as E:
#                 print("Exception", E)
#                 lora_strength = 1
#                 lora_filename = "add_detail.safetensors"
#                 pipe.load_lora_weights(
#                         f"./lora/{lora_filename.split('.')[0]}", weight_name=lora_filename)
#                 lora_prompt = "<lora:add_detail:" + \
#                     str(lora_strength) + ">, "

#             try:
#                 height = int(ast.literal_eval(data['height']))
#                 width = int(ast.literal_eval(data['width']))
#             except:
#                 height = 512
#                 width = 512
#             print(height, width, "height and width")
#             # if len(prompt:
#             prompt = lora_prompt + prompt + \
#                 " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality | hard focus, smooth, depth of field, 8K UHD"
#             # " , | ultra highly detailed | masterpiece | 8K | cinematic | focused | high quality, studio shoot, Nikon D850"
#             # compel = Compel(tokenizer=pipe.tokenizer,
#             #                 text_encoder=pipe.text_encoder)
#             # prompt_embeds = compel.build_conditioning_tensor(
#             #     prompt) if prompt else None
#             image = pipe(prompt=prompt, negative_prompt=nprompt, width=width, height=height,
#                          guidance_scale=gscale, num_inference_steps=steps, generator=generator).images[0]

#             pipe.unload_lora_weights()

#             try:
#                 # image = image.convert("RGBA")
#                 scale = 4
#                 # convert image into numpy
#                 image = np.array(image)
#                 image = upsampler.enhance(image, outscale=scale)
#                 # print(image.shape, "image shape")
#                 # print("image", image)
#             except Exception as E:
#                 print(E)

#             # low_threshold = 100
#             # high_threshold = 200

#             # # convert image into numpy
#             # image = np.array(image)

#             # image = cv2.Canny(image, low_threshold, high_threshold)
#             # image = image[:, :, None]
#             # image = np.concatenate([image, image, image], axis=2)
#             # control_image = Image.fromarray(image)

#             # print(prompt, "prompt")
#             # # try:
#             # #     height = int(ast.literal_eval(data['height']))
#             # #     width = int(ast.literal_eval(data['width']))
#             # # except:
#             # height = 1024
#             # width = 1024
#             # # print(height, width, "height and width")

#             # # image.save("sd-generated-image.jpeg")
#             # # image.save('sd-generated-image.jpeg',
#             # #            optimize=True, quality=80)
#             # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#             #                 generator=generator, image=control_image).images[0]

#         # read image as pillow and convert it to base64string image
#         elif model == 'control':
#             image = data['control_image']
#             low = int(ast.literal_eval(data['low']))
#             high = int(ast.literal_eval(data['high']))
#             guessmode = bool(data['guessmode'])

#             try:
#                 lora = str(data['lora'])

#                 if lora:
#                     lora_strength = float(data['lora_strength'])
#                     lora_filename = f"{lora}.safetensors"
#                     control.load_lora_weights(
#                         f"./lora/{lora}", weight_name=lora_filename)

#                     if lora == 'add_detail':
#                         lora_prompt = "<lora:add_detail:" + \
#                             str(lora_strength) + ">, "
#                 else:
#                     lora_prompt = ""

#             except:
#                 # lora_strength = 1
#                 # lora_filename = "add_detail.safetensors"
#                 # control.load_lora_weights(
#                 #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
#                 # lora_prompt = "<lora:add_detail:" + \
#                 #     str(lora_strength) + ">, "
#                 lora_prompt = ""
#             print("prompt 1", prompt)
#             print("lora prompt", lora_prompt)

#             # convert image from base64string to PIL image
#             image = Image.open(BytesIO(base64.b64decode(image)))
#             # image.save("control.png")
#             # print(image.size)
#             # # get the image width and height
#             # width, height = image.size
#             # if width*height > 3000*3000:
#             #     width = int(width*0.2)
#             #     height = int(height*0.2)
#             # elif width*height > 2000*2000:
#             #     width = int(width*0.4)
#             #     height = int(height*0.4)
#             # elif width*height > 1000*1000:
#             #     width = int(width*0.6)
#             #     height = int(height*0.6)
#             # elif width*height > 500*500:
#             #     width = int(width*0.8)
#             #     height = int(height*0.8)
#             # print(width, height, "new width and height")
#             # # check if height and width is divisible by 8
#             # if width % 8 != 0:
#             #     width = width - (width % 8)
#             # if height % 8 != 0:
#             #     height = height - (height % 8)

#             # image = image.resize((width, height))

#             # image = depth_estimator(image)['predicted_depth'][0]
#             # image = image.numpy()

#             # image_depth = image.copy()
#             # image_depth -= np.min(image_depth)
#             # image_depth /= np.max(image_depth)

#             # bg_threhold = 0.4

#             # x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#             # x[image_depth < bg_threhold] = 0

#             # y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
#             # y[image_depth < bg_threhold] = 0

#             # z = np.ones_like(x) * np.pi * 2.0

#             # image = np.stack([x, y, z], axis=2)
#             # image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
#             # image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
#             # image = Image.fromarray(image)
#             image = np.array(image)

#             low_threshold = low
#             high_threshold = high

#             image = cv2.Canny(image, low_threshold, high_threshold)
#             image = image[:, :, None]
#             image = np.concatenate([image, image, image], axis=2)
#             control_image = Image.fromarray(image)
#             control_image.save('sd-generated-image.jpeg',
#                                optimize=True, quality=80)
#             # change size of image
#             prompt = lora_prompt + prompt
#             print(prompt, "prompt")
#             try:
#                 height = int(ast.literal_eval(data['height']))
#                 width = int(ast.literal_eval(data['width']))
#             except:
#                 height = 512
#                 width = 512
#             print(height, width, "height and width")

#             # image.save("sd-generated-image.jpeg")
#             # image.save('sd-generated-image.jpeg',
#             #            optimize=True, quality=80)
#             # compel = Compel(tokenizer=pipe.tokenizer,
#             #                 text_encoder=pipe.text_encoder)
#             # prompt_embeds = compel.build_conditioning_tensor(
#             #     prompt) if prompt else None
#             image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#                             generator=generator, guess_mode=guessmode, image=control_image).images[0]
#             # image = control(prompt=prompt, negative_prompt=nprompt, width=width, height=height, guidance_scale=gscale, num_inference_steps=steps,
#             #                 generator=generator, image=image).images[0]
#             control.unload_lora_weights()
#             try:
#                 # image = image.convert("RGBA")
#                 scale = 4
#                 # convert image into numpy
#                 image = np.array(image)
#                 image = upsampler.enhance(image, outscale=scale)
#                 # print(image.shape, "image shape")
#                 # print("image", image)
#             except Exception as E:
#                 print(E)

#         # elif model == 'upscale':
#         #     print(model, "model")
#         #     low_res_img = data['low_res_image']
#         #     try:
#         #         target_size = int(data['target_size'])
#         #         constant = int(data['constant'])
#         #         low_res_img = Image.open(
#         #             BytesIO(base64.b64decode(low_res_img)))
#         #         low_res_img = ImageResize(low_res_img, target_size, constant)
#         #     except:
#         #         low_res_img = Image.open(
#         #             BytesIO(base64.b64decode(low_res_img)))

#         #     # print( low res image size )
#         #     print(low_res_img.size, "image size ---")

#         #     try:
#         #         lora = str(data['lora'])

#         #         if lora == "add_detail":
#         #             lora_strength = float(data['lora_strength'])
#         #             lora_filename = "add_detail.safetensors"
#         #             control.load_lora_weights(
#         #                 "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)

#         #             lora_prompt = "<lora:add_detail:" + \
#         #                 str(lora_strength) + ">, "
#         #         else:
#         #             lora_prompt = ""

#         #     except:
#         #         # lora_strength = 1
#         #         # lora_filename = "add_detail.safetensors"
#         #         # control.load_lora_weights(
#         #         #     "OedoSoldier/detail-tweaker-lora", weight_name=lora_filename)
#         #         # lora_prompt = "<lora:add_detail:" + \
#         #         #     str(lora_strength) + ">, "
#         #         lora_prompt = ""

#         #     print("prompt 1", prompt)
#         #     print("lora prompt", lora_prompt)
#         #     prompt = lora_prompt + prompt
#         #     image = upscale(prompt=prompt, negative_prompt=nprompt, guidance_scale=gscale,
#         #                     num_inference_steps=steps, generator=generator, image=low_res_img).images[0]

#         # image = Image.open('sd-generated-image.png')
#         # convert image into webp
#         image = image[0]
#         # print("image", image)
#         image = Image.fromarray(image)
#         image = image.convert("RGB")
#         # image.save("sd-generated-image.jpeg")

#         buffered = BytesIO()
#         image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         return jsonify({"image": img_str}), 200
#     except Exception as E:
#         print(E)
#         return jsonify({"message": "error something is wrong : " + str(E)}), 400




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4949)
