# Photorealistic style transfer
Unofficial Pytorch implementation of [Ultrafast Photorealistic Style Transfer via Neural Architecture Search](https://arxiv.org/abs/1912.02398). Due to the complexity of P-Step, it was not implemented. However, the original Torch implementation by the author can be found [here](https://github.com/pkuanjie/StyleNAS).

This implementation uses Nvidia DALI and AMP to accelerate the training process, with WanDB employed for monitoring.

## Prerequisites

1. Clone this repository 

   ```bash
   git clone https://github.com/jackdaw213/Photorealistic-style-transfer
   cd Photorealistic-style-transfer
   ```
2. Install Conda and create an environment
    ```shell
    conda create -n photorealistic_style_transfer python=3.12
    ```
3. Install all dependencies from requirements.txt
    ```shell
    conda activate photorealistic_style_transfer
    pip install nvidia-pyindex
    pip install -r requirements.txt
    ```
This should prepare the Conda environment for both training and testing (pretrained model available below)

## Train

1. Download the [COCO](https://github.com/nightrome/cocostuff), extract the files and organize them into the 'data' folder, with subfolders 'train' and 'val'

2. Train the model.

    ```python
    python train.py --enable_dali --enable_amp --enable_wandb
    ```

    ```
    train.py [-h]
             [--epochs EPOCHS]
             [--batch_size BATCH_SIZE]
             [--num_workers NUM_WORKERS]
             [--train_dir TRAIN_DIR_CONTENT]
             [--val_dir VAL_DIR_CONTENT]
             [--optimizer OPTIMIZER]
             [--learning_rate LEARNING_RATE]
             [--momentum MOMENTUM]
             [--resume_id RESUME_ID]
             [--checkpoint_freq CHECKPOINT_FREQ]
             [--amp_dtype AMP_DTYPE]
             [--enable_dali]
             [--enable_amp]
             [--enable_wandb]
    ```

    The model was trained on an RTX 3080 10G for 10 epoches

    | Training setup      | Batch size  | GPU memory usage | Training time |
    |---------------------|-------------|------------------|---------------|
    | DALI                | 8           | 8.3GB            | 6.7 hours     |
    | DALI + AMP          | 16          | 8.3GB            | 4 hours       |
    | DataLoader          | 16          | 9.9GB            | 6.3 hours     |
    | DataLoader + AMP    | 16          | 8.4GB            | 4.1 hours     |

    WARNING: Nvidia DALI only supports Nvidia GPUs. BFloat16 is supported only on RTX 3000/Ampere GPUs and above, while GPU Direct Storage (GDS) is supported only on server-class GPUs. Using Float16 might cause NaN loss during training, whereas BFloat16 does not.

## Test

1. Download the pretrained model [here](https://drive.google.com/file/d/1BrPzNVp-0121XEYP6jBfciM3h9Y9xUX5/view?usp=sharing) and put it in the model folder

2. Generate the output image using the command bellow.

    ```python
    python test -c content_image_path -s style_image_path
    ```

    ```
    test.py [-h] 
            [--content CONTENT] 
            [--style STYLE]
            [--model MODEL_PATH] 
    ```
    WARNING: High-resolution images will use a significant amount of memory. It is recommended to downscale the images to a reasonable resolution.

## Result

![image](https://github.com/jackdaw213/Photorealistic-style-transfer/blob/master/results/comp1.jpg)
![image](https://github.com/jackdaw213/Photorealistic-style-transfer/blob/master/results/comp2.jpg)
![image](https://github.com/jackdaw213/Photorealistic-style-transfer/blob/master/results/comp3.jpg)
![image](https://github.com/jackdaw213/Photorealistic-style-transfer/blob/master/results/comp4.jpg)
![image](https://github.com/jackdaw213/Photorealistic-style-transfer/blob/master/results/comp5.jpg)

## References

- [An, J., Xiong, H., Huan, J., & Luo, J. (2019). Ultrafast Photorealistic Style Transfer via Neural Architecture Search](hhttps://arxiv.org/abs/1912.02398)
- [Original implementation](https://github.com/pkuanjie/StyleNAS) 

