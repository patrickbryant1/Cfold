#This test case starts from the MSA.
#If you want to generate the MSA use HHblits.

#MSA feats
ID=4AVA
FASTA_DIR=./data/test/
MSA_DIR=./data/test/
OUTDIR=./data/test/

python3 ./src/make_msa_seq_feats.py --input_fasta_path $FASTA_DIR/$ID'.fasta' \
--input_msas $MSA_DIR/$ID'.a3m' --outdir $OUTDIR
