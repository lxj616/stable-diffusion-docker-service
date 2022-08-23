FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
MAINTAINER lxj616 <lxj616cn@gmail.com>

#PIP_MIRROR=""
ENV PIP_MIRROR="-i https://mirrors.ustc.edu.cn/pypi/web/simple"
RUN sed -i -e 's/archive\.ubuntu/mirrors\.163/' /etc/apt/sources.list

RUN apt update
RUN  apt install git wget -y
RUN  apt install ffmpeg libsm6 libxext6  -y
RUN  pip install albumentations opencv-python pudb imageio imageio-ffmpeg pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity transformers webdataset kornia $PIP_MIRROR
RUN  pip install sentencepiece $PIP_MIRROR
RUN  pip install redis pillow $PIP_MIRROR
RUN  pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers $PIP_MIRROR
RUN  pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip $PIP_MIRROR
RUN  pip install invisible-watermark $PIP_MIRROR
RUN  pip install diffusers $PIP_MIRROR

ADD . /workdir/stable-diffusion
RUN (cd /workdir/stable-diffusion; pip install -e .;)
RUN (cd /workdir/stable-diffusion; python pre_cache.py;)
RUN (cd /workdir/stable-diffusion; mkdir -p models/ldm/stable-diffusion-v1/;)
WORKDIR /workdir/stable-diffusion
CMD ["python","/workdir/stable-diffusion/txt2img_serve.py"]
