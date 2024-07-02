<div align="center">    
 
 # Multi-line Mathematical Expressions Recognition Based on CoMER Model


</div>

This project is based on the CoMER (Contextual Memory and External Representation) model and aims to participate in the ICPR 2024 (International Conference on Pattern Recognition) competition, focusing on the recognition of multi-line mathematical expressions.

## the use of docker images

### download docker image

1. baidu pan TODO
2. docker load -i nic-icpr.tar


### **📝 Prerequisites**

- CPU >= 4 cores
- RAM >= 64 GB
- SHM ≥ 16GB
- Disk >= 50 GB
- Docker >= 24.0.0

### **🚀 Start up** 

1. download the docker images
2. start up the container
    
    ```bash
    docker run -itd \
      --gpus all \
      --name nic-icpr \
      --shm-size=16g \
      -m 64g \
      nic-image \
      /bin/bash
    ```
    

## model train from the beginning

1. download the git repo
    
    ```bash
    git clone https://github.com/BFlameSwift/icpr
    ```
    
2. download and unzip data
    
    ```bash
    # downlaod TestA and Train data
    wget https://s2.kxsz.net/datad-public-1255000019/ICPR_2024/ICPR%202024%20Competition%20o
    n%20Multi-line%20Mathematical%20Expressions%20Recognition%20RegistrationForm%20trainning%20set.zip
    
    wget https://s2.kxsz.net/datad-public-1255000019/ICPR_2024/ICPR%202024%20Competition%20on%20Multi-line%20Mathematical%20Expressions%20Recognition%20RegistrationForm%20Test_A%20set.zip
    
    # downlaod miniconda  
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_24.3.0-0-Linux-x86_64.sh
    
    # create dir
    mkdir -p  ./data/testAdata/
    
    # unzip file 
    unzip ICPR\ 2024\ Competition\ on\ Multi-line\ Mathematical\ Expressions\ Recognition\ RegistrationForm\ Test_A\ set.zip -d   ./data/testAdata/
    unzip ICPR\ 2024\ Competition\ on\ Multi-line\ Mathematical\ Expressions\ Recognition\ RegistrationForm\ trainning\ set.zip  -d ./data/source
    ```
    
3. preprocess data
    
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
    # gpus: 4
    # accelerator: ddp
    ```
    

## model inference

1. preprocess testA data
    
    ```bash
    conda activate icpr
    
    python 4-process_testA.py
    ./5-rebag_testAdata.sh
    ```
    
2. model inference in testA data
    
    ```bash
    python ./scripts/test/test.py  1 gray
    ```
    
3. get answer.json
    
    ```bash
    python 6-process_testA_answer.py
    ```



## Project structure
```bash
├── README.md
├── comer               # model definition folder
├── config.yaml         # config for CoMER hyperparameter
├── data.zip            # train data or test data zip file
├── deploy              # data preprocess and test folder
├── eval_all.sh         # script to evaluate model on all CROHME test sets
├── example
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb   # HMER demo
├── lightning_logs      # training logs
│   └── version_0
│       ├── checkpoints
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── notebooks           # jupyter notebooks
├── scripts             # evaluation scripts
├── setup.cfg
├── setup.py
└── train.py
```

## Install dependencies   
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



## Reference
[CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)