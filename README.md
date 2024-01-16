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

# Run the test case

## Search Uniclust30 with HHblits to generate an MSA
```
ID=4AVA
FASTA_DIR=./data/test/
UNICLUST=./data/uniclust30_2018_08/uniclust30_2018_08
OUTDIR=./data/test/

./hh-suite/build/bin/hhblits -i $FASTA_DIR/$ID.fasta -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m'
```

## MSA feats
```
MSA_DIR=./data/test/
OUTDIR=./data/test/

python3 ./src/make_msa_seq_feats.py --input_fasta_path $FASTA_DIR/$ID'.fasta' \
--input_msas $MSA_DIR/$ID'.a3m' --outdir $OUTDIR
```

## Predict
```
FEATURE_DIR=./data/test/
PARAMS=./params10000.npy
OUTDIR=./data/test/

NUM_REC=3 #Increase for hard targets
NUM_SAMPLES=13 #Increase for hard targets

python3 ./src/predict_with_clusters.py --feature_dir $FEATURE_DIR \
--predict_id $ID \
--ckpt_params $PARAMS \
--num_recycles $NUM_REC \
--num_samples_per_cluster $NUM_SAMPLES \
--outdir $OUTDIR/
```
