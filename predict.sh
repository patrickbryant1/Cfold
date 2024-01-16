## Search Uniclust30 with HHblits to generate an MSA

ID=4AVA
FASTA_DIR=./data/test/
UNICLUST=./data/uniclust30_2018_08/uniclust30_2018_08
OUTDIR=./data/test/
./hh-suite/build/bin/hhblits -i $FASTA_DIR/$ID.fasta -d $UNICLUST -E 0.001 -all -oa3m $OUTDIR/$ID'.a3m'


## MSA feats
conda activate cfold
MSA_DIR=./data/test/
OUTDIR=./data/test/

python3 ./src/make_msa_seq_feats.py --input_fasta_path $FASTA_DIR/$ID'.fasta' \
--input_msas $MSA_DIR/$ID'.a3m' --outdir $OUTDIR


## Predict
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
