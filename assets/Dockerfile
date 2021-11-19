# Note that you need to put your mujoco license in mjkey.txt

FROM ubuntu:18.04

ARG INSTALL_CW_PACKAGE=true

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
        && rm mujoco.zip \
        && wget -P /opt/.mujoco/ https://roboti.us/file/mjkey.txt

RUN python3 -m pip install -U pip==20.2

RUN mkdir /continualworld
COPY continualworld/ /continualworld/continualworld
COPY examples/ /continualworld/examples
COPY /*.py /continualworld/
COPY /*.ipynb /continualworld/

RUN cd /continualworld && \
    if $INSTALL_CW_PACKAGE ; \
        then pip install -e .; \
        else python3 setup.py egg_info && pip install -r continualworld.egg-info/requires.txt; \
    fi

RUN chmod a+rw -R /usr/local/lib/python3.6
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN chmod +r /opt/.mujoco/mjkey.txt

WORKDIR /continualworld
