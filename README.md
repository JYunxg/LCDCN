# LCDCN
Local Community Detection based on Core Node with Deep Feature Fusion 
## Requirements
Python = 3.7, PyTorch == 1.8.1, torchvision == 0.9.1
## Dataset
Dataset and pre-trained models can be downloaded here 

To run code successfully, please unzip the dataset and place it in the current directory. 
## Pre-train DLAE and GAAE models
python preae.py --name acm --epochs 50 --n_clusters 3
python pregae.py --name acm --epochs 50 --n_cluters 3
## Train our LCDCDN model
python dgc_efr.py --name acm --epochs 200
