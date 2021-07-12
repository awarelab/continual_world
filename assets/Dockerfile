# Note that you need to put your mujcoco license in mjkey.txt
# Note that you need an insitutation type license when working with docker
# http://www.mujoco.org/forum/index.php?threads/getid_linux-outputs-different-machine-id-in-docker-container.3848/#:~:text=Hardware-locked%20activation%20keys%20do%20not%20work%20with%20Docker.,has%20unlocked%20keys.%20Emo%20Todorov%2C%20Oct%2031%2C%202018

FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ENV  LANG=C.UTF-8
ENV  DEBIAN_FRONTEND=noninteractive
ENV  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/.mujoco/mujoco200/bin
ENV  MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200/
ENV  MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt

RUN apt-get -y update -q
RUN apt-get -y install git ffmpeg python3-pip curl libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
               libosmesa6-dev software-properties-common net-tools unzip vim wget xpra xserver-xorg-dev git virtualenv

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
RUN chmod +x /usr/local/bin/patchelf

RUN mkdir -p /opt/.mujoco \
        && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
        && unzip mujoco.zip -d /opt/.mujoco \
        && mv /opt/.mujoco/mujoco200_linux /opt/.mujoco/mujoco200 \
        && rm mujoco.zip

COPY mjkey.txt /opt/.mujoco/

# PM (6-7-2021) I put it to force the pip version.
RUN python3 -m pip install -U pip==20.2
COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

RUN chmod a+rw -R /usr/local/lib/python3.6
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN chmod +r /opt/.mujoco/mjkey.txt
