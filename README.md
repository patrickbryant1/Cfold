# Cfold

**Structure prediction of alternative protein conformations**

<img src="./Logo.svg"/>



\
Cfold is a structure prediction network similar to AlphaFold2 that is trained on a conformational split of the PDB. Cfold is designed for predicting alternative conformations of protein structures. [Read more about it in the paper here](https://www.biorxiv.org/content/10.1101/2023.09.25.559256v1)
\
\
AlphaFold2 is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) and so is Cfold, which is a derivative thereof. The Cfold parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).
\
\
**You may not use these files except in compliance with the licenses.**

# Colab (run in the web)

[Colab Notebook](https://colab.research.google.com/github/patrickbryant1/Cfold/blob/master/Cfold.ipynb)

# Local installation

The entire installation takes <1 hour on a standard computer. \
The runtime will depend on the GPU you have available, the size of the protein
you are predicting and the number of samples taken. On an NVIDIA A100 GPU, the
prediction time is a few minutes per sample for a protein of a few hundred amino acids.

We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \

First install miniconda, see: https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html

```
bash install_dependencies.sh
```
If the conda doesn't work for you - see "pip_pkgs.txt"

# Run the test case
## (a few minutes)
```
bash predict.sh
```

# Data availability
https://zenodo.org/records/10837082

# Citation
Bryant P. Structure prediction of alternative protein conformations. bioRxiv. 2023. p. 2023.09.25.559256. doi.org/10.1101/2023.09.25.559256
