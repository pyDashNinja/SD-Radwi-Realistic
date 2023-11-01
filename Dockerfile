FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


RUN apt-get update && \
    apt-get install -y python3 python3-pip git ffmpeg libsm6 libxext6 libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000


ENTRYPOINT ["python", "handler.py"]