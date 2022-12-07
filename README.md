# Characterizing Datapoints via Second-Split Forgetting

Repository for the paper [Characterizing Datapoints via Second-Split Forgetting](https://openreview.net/forum?id=yKDKNzjHg8N) by [Pratyush Maini](https://pratyushmaini.github.io), [Saurabh Garg](https://saurabhgarg1996.github.io/), [Zachary Lipton](https://www.zacharylipton.com/) and [Zico Kolter](https://zicokolter.com/). This work was accepted at [NeurIPS 2022](https://neurips.cc/Conferences/2022/).


## What does this repository contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found below.

## Dependencies
The repository is written using `python 3.10`. To install dependencies run the command:

`conda create -n ssft python=3.10`    
`conda activate ssft`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
`pip install matplotlib`


## Datasets
You can download preprocessed datasets used for experiments with GANs (CIFAR5m and CIFAR10_DCGAN), and on Imagenette dataset from [this link](https://drive.google.com/drive/folders/15D9OXUmlqpCXBG7ldr_UAiP72_dAmKyh?usp=share_link)
 
## Play Along Jupyter Notebook
The easiest way to get started is to play with the Jupyter notebook provided at `main.ipynb`. Through this, you can run all of the experiments performed in our work by modifying the names of the dataset and other hyperpaprameters listed therein.


## Model Training and Logging
To run experiments with multiple seeds through a python script, you can use the following command
`python train.py --dataset1 mnist --noise_1 0.1 --model_type resnet9`


## Improve Dataset Quality
`python remove_examples.py --dataset1 cifar10dcgan --removal_metric ssft`


## Rare-group Simulation
We simulate a dataset with long tails of rare subpopulations by taking a subset of the CIFAR100 dataset. For each of the 20 superclassess, we sample examples from the 5 sub-groups with an exponentially decaying frequency to create a 20-class classification problem. Finally, we randomize these results over multiple different orderings of these subgroups.

`python cifar100_superclass.py \
--dataset1 cifar100 \
--weight_decay 5e-4 \
--model resnet9 \
--batch_size 512 \
--lr1 0.1 \
--lr2 0.05 \
--log_factor 2 \
--noise1 0.1 \
--noise2 0.1 \
--sched triangle`

