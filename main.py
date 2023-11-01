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

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
request_queue = "request_queue"  # Name of the Redis queue

app = Flask(__name__)
CORS(app)
# allow all origins
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}

@app.before_request
@before_first_request
def load_model():

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




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4949)
