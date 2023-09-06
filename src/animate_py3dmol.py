import py3Dmol

import sys
import os
import pdb

import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description = '''Visualise sampled conformations''')
parser.add_argument('--pdbdir', nargs=1, type= str, required=True, help = "Path to directory with PDB files")
parser.add_argument('--order_df', nargs=1, type= str, required=True, help = "Path to csv with order of PDB files")
parser.add_argument('--outdir', nargs=1, type= str, help = 'Outdir.')


#################FUNCTIONS#################

class Atom(dict):
    def __init__(self, line):
        self["type"] = line[0:6].strip()
        self["idx"] = line[6:11].strip()
        self["name"] = line[12:16].strip()
        self["resname"] = line[17:20].strip()
        self["resid"] = int(int(line[22:26]))
        self["x"] = float(line[30:38])
        self["y"] = float(line[38:46])
        self["z"] = float(line[46:54])
        self["sym"] = line[76:78].strip()

    def __str__(self):
        line = list(" " * 80)

        line[0:6] = self["type"].ljust(6)
        line[6:11] = self["idx"].ljust(5)
        line[12:16] = self["name"].ljust(4)
        line[17:20] = self["resname"].ljust(3)
        line[22:26] = str(self["resid"]).ljust(4)
        line[30:38] = str(self["x"]).rjust(8)
        line[38:46] = str(self["y"]).rjust(8)
        line[46:54] = str(self["z"]).rjust(8)
        line[76:78] = self["sym"].rjust(2)
        return "".join(line) + "\n"

class Molecule(list):
    def __init__(self, file):
        for line in file:
            if "ATOM" in line or "HETATM" in line:
                self.append(Atom(line))

    def __str__(self):
        outstr = ""
        for at in self:
            outstr += str(at)

        return outstr





#################MAIN####################

#Parse args
args = parser.parse_args()
#Get data
pdbdir = args.pdbdir[0]
order_df = pd.read_csv(args.order_df[0]).loc[:10]
outdir = args.outdir[0]

molecules = []
for ind,row in order_df.iterrows():
    with open(pdbdir+row['name']) as ifile:
        molecules.append(Molecule(ifile))


view = py3Dmol.view(width=400, height=300)

models = ""
for i, mol in enumerate(molecules):
    models += "MODEL " + str(i) + "\n"
    models += str(mol)
    models += "ENDMDL\n"
view.addModelsAsFrames(models)

for i, at in enumerate(molecules[0]):
    default = {"cartoon": {'color': 'grey'}}
    view.setStyle({'model': -1, 'serial': i+1}, at.get("pymol", default))

view.zoomTo()
view.animate({'loop': 'backAndForth'})
view.show()
