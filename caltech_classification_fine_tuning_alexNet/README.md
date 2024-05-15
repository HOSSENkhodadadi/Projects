# Caltech 101 Classification
## Description
This project implements a deep learning model for image classification on the Caltech 101 dataset. The model is trained using PyTorch and monitored using Weights & Biases (wandb).

## Installation
To run this project, you need to install the Weights & Biases library (wandb). You can install it via pip:<br/>
pip install wandb

You also need to clone the dataset repository. Run the following command in your terminal:<br/>
git clone https://github.com/MachineLearning2020/Homework2-Caltech101.git



## Usage
Once you have installed the required libraries and cloned the dataset repository, you can use the following command-line arguments to customize the training process:

python caltech_101_kaggle_v4.py --epoch NUM_EPOCHS --batch_size BATCH_SIZE --lr LR --momentum MOMENTUM --weight_decay WEIGHT_DECAY --fine_tune_mode PRETRAINED --fine_tune_setting FINE_TUNE_SETTING

- `--epoch`: Number of epochs for training (default is 30).
- `--batch_size`: Batch size for training (default is 16).
- `--lr`: Learning rate (default is 0.001).
- `--momentum`: Momentum value (default is 0.9).
- `--weight_decay`: Weight decay (default is 1e-5).
- `--fine_tune_mode`: Boolean value indicating whether to use fine-tuning or not (default is True).
- `--fine_tune_setting`: Fine-tuning setting (default is "all").you may choose this parameter among `all`, `classifier` and `features`  

Note: You can omit any arguments to use their default values.

## Example
Here's an example command to start training:

python caltech_101_kaggle_v4.py --epoch 10 --batch_size 32 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --fine_tune_mode True --fine_tune_setting "setting_1"

This command will train the model for 10 epochs with a batch size of 32, learning rate of 0.001, momentum of 0.9, weight decay of 0.0001, using fine-tuning mode with setting "setting_1".

## Acknowledgments
- This project uses the Caltech 101 dataset.
- Weights & Biases (wandb) is used for experiment tracking and visualization.


