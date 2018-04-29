# PopPhy-CNN

PopPhy-CNN,a novel convolutional neural networks (CNN) learning architecture that effectively exploits phylogentic structure in microbial taxa. PopPhy-CNN provides an input format of 2D matrix created by embedding the phylogenetic tree that is populated with the relative abundance of microbial taxa in a metagenomic sample. This conversion empowers CNNs to explore the spatial relationship of the taxonomic annotations on the tree and their quantitative characteristics in metagenomic data.


## Publication:
* Derek Reiman, Ahmed A. Metwally, Yang Dai. "PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolution Neural Networks for Metagenomic Data", bioRxiv, 2018.  [[paper](https://www.biorxiv.org/content/early/2018/01/31/257931)]

## Execution:

### Prerequisites
  - Python 2.7.14
  - Libraries: `pip install theano numpy pandas joblib xmltodict untangle sklearn network`
  
### To train the network on the *Cirrhosis* dataset
Assuming that you are in the project's root dir.
```bash
cd data/Cirrhosis/
python ./parse_data.py
cd ../../src/
python ./prepare_data.py -r
python ./train.py
``` 