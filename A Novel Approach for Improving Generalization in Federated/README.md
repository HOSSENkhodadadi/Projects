# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Starting code for the Federated Learning project. Some functions are explicitly left blank for students to fill in.

## Setup
#### Environment
If not working on CoLab, install environment with conda (preferred): 
```bash 
conda env create -f mldl23fl.yml
```

#### Datasets
The repository supports experiments on the following datasets:
1. **FEMNIST** (Federated Extended MNIST) from LEAF benchmark [1]
   - Task: image classification on 62 classes
   - 3,500 users
   - Instructions for download and preprocessing in ```data/femnist/``` 

## How to run
The ```load-main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of FedAvg experiment (**NB** training hyperparameters need to explicitly specified by the students):

- **FEMNIST** (Image Classification)
```bash
python main.py --dataset femnist --model resnet18 --num_rounds 1000 --num_epochs 5 --clients_per_round 10 
```

## References
[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." Workshop on Federated Learning for Data Privacy and Confidentiality (2019). 


