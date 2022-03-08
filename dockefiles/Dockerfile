FROM ubuntu:18.04

MAINTAINER Matteo Perotto <perotto.matteo@gmail.com>

# ======================================
# Set up some basic packages
# ======================================

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
    python3-venv
    
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

RUN python3 -m venv env 
RUN . env/bin/activate && pip install --upgrade pip && pip install numpy matplotlib  

ADD ./entrypoint.sh 
ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
