FROM ubuntu:18.04

MAINTAINER Matteo Perotto <perotto.matteo@gmail.com>

# ======================================
# Set up some basic packages
# ======================================

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends \
    apt-utils \
    sudo \
    nano \
    curl \
    wget \
    unzip \
    locales \
    software-properties-common 
# ======================================
# Set up locales
# ======================================

RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/g' /etc/locale.gen && \
    locale-gen &&\
    update-locale LANG="en_US.UTF-8"

# ======================================
# Set up development packages
# ======================================

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gdb \
    cmake \
    cmake-curses-gui 

RUN apt-get install -y \
    python3\
    python3-pip \
    python3-venv \
    python3-tk

RUN apt-get install -y \
    qt5-default
     
# ======================================
# Create user and switch to it 
# ======================================
RUN useradd -l -G sudo -md /home/user -s /bin/bash -p user user && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers
USER user

# ======================================
# Create virtual environment 
# ======================================
WORKDIR /home/user 

RUN git clone https://github.com/MatteoPerotto/GPIS.git
RUN python3 -m venv env 
RUN . env/bin/activate && pip install --upgrade pip && pip install numpy matplotlib pyqt5 sklearn pymeshlab open3d 

ADD ./entrypoint.sh /home/user
ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
