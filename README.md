# Autoencoders-in-Spike-Sorting
Autoencoders, a type of neural network that allow for unsupervised learning, can be used in the feature extraction of spike sorting.

This study has been published in PLOS One:
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282810
- DOI: 10.1371/journal.pone.0282810

## Citation
We would appreciate it if you cite the paper when you use this work:

- For Plain Text:
```
E.-R. Ardelean, A. Coporîie, A.-M. Ichim, M. Dînșoreanu, and R. C. Mureșan, “A study of autoencoders as a feature extraction technique for spike sorting,” PLOS ONE, vol. 18, no. 3, p. e0282810, Mar. 2023, doi: 10.1371/journal.pone.0282810.
```

## Setup
The 'requirements.txt' file indicates the dependencies required for running the code. 

The synthetic data used in this study can be downloaded from: 
https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO
or
https://www.kaggle.com/datasets/ardeleanrichard/simulationsdataset.

The real data used in this study can be downloaded from:
https://www.kaggle.com/datasets/ardeleanrichard/realdata
or in the 'real_data' folder of the repository.


In the constants.py file the path to the DATA folder can be set. We recommend the following structure for the data:

DATA/
* TINS/
  * M045_009/ : insert the real data files
* SIMULATIONS/ : insert the synthetic data files


# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)
