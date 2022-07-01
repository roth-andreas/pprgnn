# Personalized Page Rank Graph Neural Network

This is the official implementation of the Personalized Page Rank Graph Neural Network (PPRGNN) with pytorch geometric.
To execute the experiments run the corresponding main_<NAME>.py file.

### Required Libraries:

* pytorch
* pytorch_geometric


### Datasets

* PPI, MUTAG, PROTEINS, NCI1, COX2 will be downloaded automatically via pytorch_geometric
* Amazon and PTC. Please download the dataset/ folder from the dropbox link below and place the folder inside this project
  * https://www.dropbox.com/sh/rjf9nlr94c7zmuk/AAAcCRLmgBNyp4aAyKcKHYi2a?dl=0

Note: Due to the non-deteminism of *torch.spmm* repeatedly executing our code leads to slightly different results.
