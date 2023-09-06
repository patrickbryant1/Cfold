import sys
import os
import pdb

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from Bio.SVDSuperimposer import SVDSuperimposer
import shutil
import glob
import argparse


parser = argparse.ArgumentParser(description = '''Structurally align all files in a directory to a reference and write new roto-translated pdb files''')
parser.add_argument('--pdbdir', nargs=1, type= str, required=True, help = "Path to directory with PDB files")
parser.add_argument('--outdir', nargs=1, type= str, help = 'Outdir.')


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


def write_pdb(pdb_info, new_coords, name):
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

#Parse args
args = parser.parse_args()
#Get data
pdbdir = args.pdbdir[0]
outdir = args.outdir[0]

#Read in all coords
pdb_info = {'name':[], 'file_contents':[], 'all_coords':[], 'ca_coords':[]}
print('Reading preds...')
for name in glob.glob(pdbdir+'*.pdb'):
    current_info, current_coords, current_ca_coords = read_pdb(name)
    pdb_info['name'].append(name)
    pdb_info['file_contents'].append(current_info)
    pdb_info['all_coords'].append(current_coords)
    pdb_info['ca_coords'].append(current_ca_coords)

#Score all
print('Scoring...')
score_matrix, order = score_ca_diff(pdb_info['ca_coords'])
#Align according to order and write to out
#Copy the first one to out
print('Writing aligned structures...')
shutil.copy(pdb_info['name'][order[0]], outdir+pdb_info['name'][order[0]].split('/')[-1])
tr_current_ca = pdb_info['ca_coords'][order[0]]
ordered_names = [pdb_info['name'][order[0]].split('/')[-1]]
for i in range(1,len(order)):
    #Align i+1 to i
    tr_current_coords, tr_current_ca = align_coords_transform(tr_current_ca, pdb_info['ca_coords'][order[i]], pdb_info['all_coords'][order[i]])
    #Write new pdb file
    write_pdb(pdb_info['file_contents'][order[i]], tr_current_coords, pdb_info['name'][order[i]].split('/')[-1])
    ordered_names.append(pdb_info['name'][order[i]].split('/')[-1])

#Save order
order_df = pd.DataFrame()
order_df['name'] = ordered_names
order_df.to_csv(outdir+'aligned_order.csv', index=None)
