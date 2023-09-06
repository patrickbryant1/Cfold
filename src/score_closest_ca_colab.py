import sys
import os
import pdb

import pandas as pd
import numpy as np
from collections import defaultdict
from Bio.SVDSuperimposer import SVDSuperimposer
import shutil
import glob
import argparse


#################FUNCTIONS#################


def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def read_pdb(pdbfile):
    '''Read a pdb file per chain
    '''
    pdb_file_info = []
    all_coords = []
    CA_coords = []

    with open(pdbfile) as file:
        for line in file:
            if line.startswith('ATOM'):
                #Parse line
                record = parse_atm_record(line)
                #Save line
                pdb_file_info.append(line.rstrip())
                #Save coords
                all_coords.append([record['x'],record['y'],record['z']])
                if record['atm_name']=='CA':
                    CA_coords.append([record['x'],record['y'],record['z']])


    return pdb_file_info, np.array(all_coords), np.array(CA_coords)

def score_ca_diff(all_ca_coords):
    """Score all CA coords against each other
    """

    num_samples = len(all_ca_coords)
    score_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dmat_i = np.sqrt(np.sum((all_ca_coords[i][:,None]-all_ca_coords[i][None,:])**2,axis=-1))
        score_mat[i,i]=1000
        for j in range(i+1, num_samples):
            dmat_j = np.sqrt(np.sum((all_ca_coords[j][:,None]-all_ca_coords[j][None,:])**2,axis=-1))
            #Diff
            diff = np.mean(np.sqrt((dmat_i-dmat_j)**2))
            score_mat[i,j] = diff
            score_mat[j,i] = diff

    #Establish the order from the score mat
    order = []
    min_ind = np.unravel_index(score_mat.argmin(), score_mat.shape)
    all_inds = np.arange(num_samples)
    order.append(min_ind[0])
    ci = min_ind[1]
    while len(order)<num_samples:
        row_ci = score_mat[ci]
        order.append(ci)
        #Get non-fetched samples
        remaining = np.setdiff1d(all_inds, order)
        if len(remaining)==0:
            break
        ci = remaining[np.argmin(row_ci[remaining])]

    return score_mat, order

def align_coords_transform(ref_ca, current_ca, current_coords):
    """Align CA coords and roto-translate all afterwards
    """

    sup = SVDSuperimposer()

    #Set the coordinates to be superimposed.
    #coords will be put on top of reference_coords.
    sup.set(ref_ca, current_ca) #(reference_coords, coords)
    sup.run()
    rot, tran = sup.get_rotran()

    #Rotate coords from new chain to its new relative position/orientation
    tr_current_coords = np.dot(current_coords, rot) + tran
    tr_current_ca = np.dot(current_ca, rot) + tran

    return tr_current_coords, tr_current_ca


def write_pdb(pdb_info, new_coords, name, outdir):
    """Write PDB
    """


    #Open file
    outname = outdir+name
    atm_no=0
    with open(outname, 'w') as file:
        for i in range(len(pdb_info)):
            line = pdb_info[i]
            #Update line with new coords
            x,y,z = new_coords[i]
            x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
            x =' '*(8-len(x))+x
            y =' '*(8-len(y))+y
            z =' '*(8-len(z))+z
            #Write
            file.write(line[:30]+x+y+z+line[54:]+'\n')




#################MAIN####################
