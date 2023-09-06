import json
import os
import warnings
import pathlib
import pickle
import random
import sys
import time
from typing import Dict, Optional
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
#Silence tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.config.set_visible_devices([], 'GPU')

import argparse
import pandas as pd
import numpy as np
from collections import Counter
from scipy.special import softmax
import pdb

#Data loading
from tinyloader import DataLoader

#AlphaFold imports
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import features
from alphafold.model import modules

#JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run.
#This prevents this
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


##############FUNCTIONS##############
##########INPUT DATA#########

def process_features(raw_features, config, random_seed):
    """Processes features to prepare for feeding them into the model.

    Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.

    Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
    """
    return features.np_example_to_features(np_example=raw_features,
                                            config=config,
                                            random_seed=random_seed)


def load_input_feats(id, feature_dir, config, num_clusters):
    """
    Load all input feats.
    """

    #Load raw features
    msa_feature_dict = np.load(feature_dir+id+'/msa_features.pkl', allow_pickle=True)
    #Process the features on CPU (sample MSA)
    #Set the config to determine the number of clusters
    config.data.eval.max_msa_clusters = num_clusters
    #processed_feature_dict['msa_feat'].shape = num_clusts, L, 49
    processed_feature_dict = process_features(msa_feature_dict, config, np.random.choice(sys.maxsize))

    return processed_feature_dict



##########MODEL#########

def predict(config,
          feature_dir,
          predict_ids,
          num_recycles,
          num_samples_per_cluster,
          ckpt_params=None,
          outdir=None):
    """Predict a structure
    """

    #Does the config have to be updated here?
    #No - the clusters can be changed.
    #Define the forward function
    def _forward_fn(batch):
        '''Define the forward function - has to be a function for JAX
        '''
        model = modules.AlphaFold(config.model)

        return model(batch,
                    is_training=False,
                    compute_loss=False,
                    ensemble_representations=False,
                    return_representations=True)

    #The forward function is here transformed to apply and init functions which
    #can be called during training and initialisation (JAX needs functions)
    forward = hk.transform(_forward_fn)
    apply_fwd = forward.apply
    #Get a random key
    rng = jax.random.PRNGKey(42)

    for id in predict_ids:
        for num_clusts in [16, 32, 64, 128, 256, 512, 1024, 5120]:
            for i in range(num_samples_per_cluster):
                if os.path.exists(outdir+'/'+id+'_'+str(num_clusts)+'_'+str(i)+'_pred.pdb'):
                    print('Prediction',num_clusts, i+1, 'exists...')
                    continue

                #Load input feats
                batch = load_input_feats(id, feature_dir, config, num_clusts)
                for key in batch:
                    batch[key] = np.reshape(batch[key], (1, *batch[key].shape))

                batch['num_iter_recycling'] = [num_recycles]
                ret = apply_fwd(ckpt_params, rng, batch)
                #Save structure
                save_feats = {'aatype':batch['aatype'], 'residue_index':batch['residue_index']}
                result = {'predicted_lddt':ret['predicted_lddt'],
                    'structure_module':{'final_atom_positions':ret['structure_module']['final_atom_positions'],
                    'final_atom_mask': ret['structure_module']['final_atom_mask']
                    }}
                save_structure(save_feats, result, id+'_'+str(num_clusts)+'_'+str(i), outdir)



def save_structure(save_feats, result, id, outdir):
    """Save prediction

    save_feats = {'aatype':batch['aatype'][0][0], 'residue_index':batch['residue_index'][0][0]}
    result = {'predicted_lddt':aux['predicted_lddt'],
            'structure_module':{'final_atom_positions':aux['structure_module']['final_atom_positions'][0],
            'final_atom_mask': aux['structure_module']['final_atom_mask'][0]
            }}
    save_structure(save_feats, result, step_num, outdir)

    """
    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    # Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(result['predicted_lddt']['logits']) * bin_centers[None, :], axis=-1)
    plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=save_feats, result=result,  b_factors=plddt_b_factors)
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir+'/', id+'_pred.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb)



##################MAIN#######################

# #Predict
# predict(config.CONFIG,
#             feature_dir,
#             [predict_id],
#             num_recycles,
#             num_samples_per_cluster,
#             ckpt_params=ckpt_params,
#             outdir=outdir)
