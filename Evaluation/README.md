## Evaluation

### Inpainting

### Lama
#### Environment setup

1. Python virtualenv:

    ```
    virtualenv inpenv --python=/usr/bin/python3
    source inpenv/bin/activate
    pip install torch==1.8.0 torchvision==0.9.0
    
    cd lama
    pip install -r requirements.txt 
    ```

2. Conda
    
    ```
    % Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    $HOME/miniconda/bin/conda init bash

    cd lama
    conda env create -f conda_env.yml
    conda activate lama
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    ```

#### Inference <a name="prediction"></a>

Run
```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

**1. Download pre-trained models**

The best model (Places2, Places Challenge):
    
```    
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
```

All models (Places & CelebA-HQ):

```
download [https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link]
unzip lama-models.zip
```

**2. Prepare images and masks**

Download test images:

```
unzip LaMa_test_images.zip
```

OR prepare your data:
1) Create masks named as `[images_name]_maskXXX[image_suffix]`, put images and masks in the same folder. 

- You can use the [script](https://github.com/advimman/lama/blob/main/bin/gen_mask_dataset.py) for random masks generation. 
- Check the format of the files:
    ```    
    image1_mask001.png
    image1.png
    image2_mask001.png
    image2.png
    ```

2) Specify `image_suffix`, e.g. `.png` or `.jpg` or `_input.jpg` in `configs/prediction/default.yaml`.


**3. Predict**

On the host machine:

    python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

**OR** in the docker
  
The following command will pull the docker image from Docker Hub and execute the prediction script
```
bash docker/2_predict.sh $(pwd)/big-lama $(pwd)/LaMa_test_images $(pwd)/output device=cpu
```
Docker cuda:
```
bash docker/2_predict_with_gpu.sh $(pwd)/big-lama $(pwd)/LaMa_test_images $(pwd)/output
```

**4. Predict with Refinement**

On the host machine:

    python3 bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

