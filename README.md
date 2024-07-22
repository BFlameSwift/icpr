<div align="center">    
 
 # Multi-line Mathematical Expressions Recognition Based on CoMER Model

</div>
# **Multi-line Mathematical Expressions Recognition Based on CoMER Model**

This repository is based on the CoMER (Contextual Memory and External Representation) model and aims to participate in the [ICPR 2024 (International Conference on Pattern Recognition Competition](https://note.kxsz.net/share/f4025d8b-7b50-4034-ad96-1b35633861d4), focusing on the recognition of multi-line mathematical expressions.

Our team name is **None Information Given(NIG)**, winning 4th place in this competition
## **The use of docker images**

### **Download docker image**

1. [BaiduDisk with code icpr](https://pan.baidu.com/s/1I5qSFTGRyrSRPSlxcp3JkQ?pwd=icpr)
2. `docker load -i nic-image.tar`

### **📝 Prerequisites**

- CPU >= 4 cores
- RAM >= 64 GB
- SHM ≥ 16GB
- Disk >= 50 GB
- Docker >= 24.0.0

### **🚀 Start up**

1. download the docker images
2. load docker image
    
    `docker load -i nic-image.tar`
    
3. start up the container
    
    ```bash
    docker run -itd \
      --gpus all \
      --name nic-icpr \
      --shm-size=16g \
      -m 64g \
      nic-image \
      /bin/bash
    ```
    
4. `docker exec -it [id]  /bin/bash`
5. `cd /root/icpr`

## **Model train from the beginning**

1. Download the git repo
    
    ```bash
    
    git clone https://github.com/BFlameSwift/icpr
    
    ```
    
2. Download and unzip data
    
    ```bash
    # downlaod Train data
    wget https://s2.kxsz.net/datad-public-1255000019/ICPR_2024/ICPR%202024%20Competition%20o
    n%20Multi-line%20Mathematical%20Expressions%20Recognition%20RegistrationForm%20trainning%20set.zip
    
    # downlaod miniconda
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
    
    # create dir
    mkdir -p  ./data/source/
    
    # unzip test data
    unzip ICPR\ 2024\ Competition\ on\ Multi-line\ Mathematical\ Expressions\ Recognition\ RegistrationForm\ Test_A\ set.zip -d   ./data/testAdata/
    
    # unzip train data
    unzip ICPR\ 2024\ Competition\ on\ Multi-line\ Mathematical\ Expressions\ Recognition\ RegistrationForm\ trainning\ set.zip  -d ./data/source
    
    ```
    
3. Preprocess data
    
    ```bash
    cd icpr
    
    conda activate icpr
    
    cd deploy
    
    python 1-preprocess image.py
    python 2-split_train_test.py
    ./3-rebag_data.sh
    
    ```
    
4. Next, navigate to icpr folder and run `train.py`. It may take 48 hours on 8 NVIDIA 2080Ti gpus using ddp.
    
    ```bash
    # train CoMER(Fusion) model using 4 gpus and ddp
    python train.py --config config.yaml
    
    ```
    
    For single gpu user, you may change the `config.yaml` file to
    
    ```bash
    gpus: 1
    # gpus: 4# accelerator: ddp
    ```
    

## **Model inference(must)**

1. Preprocess testA data
    - download and unzip test dat
    
    ```bash
    cd icpr
    # download test data
    wget https://s2.kxsz.net/datad-public-1255000019/ICPR_2024/ICPR%202024%20Competition%20on%20Multi-line%20Mathematical%20Expressions%20Recognition%20RegistrationForm%20Test_A%20set.zip
    # create dir
    mkdir -p  ./data/testAdata/
    # unzip test data
    unzip ICPR\ 2024\ Competition\ on\ Multi-line\ Mathematical\ Expressions\ Recognition\ RegistrationForm\ Test_A\ set.zip -d   ./data/testAdata/
    ```
    
    ```bash
    conda activate icpr
    
    cd deploy
    
    python 4-process_testA.py
    
    # rebag testAdata to data.zip
    ./5-rebag_testAdata.sh
    ```
    
2. Model inference in testA data，
    
    ```bash
    # in Project root dir
    #evaluate model in lightning_logs/version_1 on test set（data.zip/gray）
    python ./scripts/test/test.py  1  gray
    ```
    
3. Get answer.json
    
    ```bash
    python 6-process_testA_answer.py
    ```
    

## **Project structure**

```bash
├── README.md
├── comer# model definition folder
├── config.yaml# config for CoMER hyperparameter
├── data.zip# train data or test data zip file
├── deploy# data preprocess and test folder
├── eval_all.sh# script to evaluate model on all CROHME test sets
├── lightning_logs# training logs
│   └── version_0
│       ├── checkpoints
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── notebooks# jupyter notebooks
├── scripts# evaluation scripts
├── setup.cfg
├── setup.py
└── train.py

```

## **Install dependencies**

```bash
cd icpr
# install project
conda create -y -n icpr python=3.7
conda activate CoMER
conda install pytorch=1.8.1 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
pip install torchvision==0.2.2
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .

```

## **Reference**

[CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)
