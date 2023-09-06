import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read,write
from ase import Atoms, Atom
import ase
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import pymatgen as mg

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data # torchtext.data 임포트
from torchtext.data import Iterator
from torch.utils.data import Dataset, DataLoader


import csv
import pandas as pd

# from data import CIFData, AtomCustomJSONInitializer, GaussianDistance
import os
import csv
import random

def get_train_val_test_loader(dataset, train_ratio=0.7, batch_size=50 ,valid_ratio=0.15, test_ratio=0.15, num_workers=1):
    
    total_size = len(dataset)
    indices = list(range(total_size))
    # train_ratio = 0.7
    # valid_ratio = 0.15
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = int(test_ratio * total_size)

    train_sampler =SubsetRandomSampler(indices[:train_size])
    valid_sampler = SubsetRandomSampler(indices[-(valid_size + test_size):-test_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size = batch_size,sampler=train_sampler, num_workers= num_workers)
    val_loader = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers )
    test_loader = DataLoader(dataset, batch_size= batch_size, sampler = test_sampler, num_workers=num_workers )

    return train_loader, val_loader, test_loader

# class remove_adsorbates(atoms):
#     self.atoms = atoms
    
#     def get_removed_atoms(self, atoms)
#         binding_sites = []
#         adsorbates_index = []

#         for atom in atoms:
#             if atom.tag == 1:
#                 binding_sites.append([atom.symbol,list(atom.position)])
#                 adsorbates_index.append(atom.index)
#         del atoms[adsorbates_index]
#         bare_slab = atoms.copy()

#         return binding_sites, bare_slab

# class get_nearest_atoms(atoms):
#     # view(atoms)
#     binding_sites, slab = remove_adsorbates(atoms)
#     binding_sites.sort(key = lambda x: x[1][2] )    
    
#     copied_atom = slab.copy()
#     copied_atom = copied_atom.repeat((2,2,1))
#     copied_atom += Atom(binding_sites[0][0],binding_sites[0][1],tag =1 )
#     ads_index = np.where((copied_atom.get_tags()) ==1)[0][0]
    
#     structure = AseAtomsAdaptor.get_structure(copied_atom)
#     # nn = structure.get_neighbors(site=structure[ads_index] , r= min(structure.lattice.abc))
#     nn = structure.get_neighbors(site=structure[ads_index] , r= 10)
#     nn.sort(key = lambda x : x[1]) # sort nearest atoms
#     nn_index = [nn[i][2] for i in range(len(nn))]
    
    
    # view(copied_atom)
    return copied_atom, nn_index

class get_atom_property(Atoms):
    def __init__(self,atoms):
        self.atoms = atoms
    
    def remove_adsorbates(self, atoms):
        binding_sites = []
        adsorbates_index = []

        for atom in atoms:
            if atom.tag == 1:
                binding_sites.append([atom.symbol,list(atom.position)])
                adsorbates_index.append(atom.index)
        del atoms[adsorbates_index]
        bare_slab = atoms.copy()
        return binding_sites, bare_slab
    
    def get_nearest_atoms(self, atoms):
        binding_sites, slab = self.remove_adsorbates(atoms)
        binding_sites.sort(key = lambda x: x[1][2] )    

        copied_atom = slab.copy()
        copied_atom = copied_atom.repeat((2,2,1))
        copied_atom += Atom(binding_sites[0][0],binding_sites[0][1],tag =1 )
        ads_index = np.where((copied_atom.get_tags()) ==1)[0][0]

        structure = AseAtomsAdaptor.get_structure(copied_atom)
        # nn = structure.get_neighbors(site=structure[ads_index] , r= min(structure.lattice.abc))
        nn = structure.get_neighbors(site=structure[ads_index] , r= 10)
        nn.sort(key = lambda x : x[1]) # sort nearest atoms
        nn_index = [nn[i][2] for i in range(len(nn))]
        return copied_atom, nn_index

    
    def block_to_num(self, block):
        """
        Convert blokc to number

        Args:
            block (str) : 's', 'p', 'd' or 'f'

        Return:
            int : 1, 2, 3, 4
        """

        if block == 's':
            return 1
        elif block == 'p':
            return 2
        elif block == 'd':
            return 3
        elif block == 'f':
            return 4
        
    def get_item(self):
        feature_df = pd.DataFrame(columns=[f'feat{i}' for i in range(10)])
        atom, nn_index = self.get_nearest_atoms(atoms)
        elements = [atom[i].symbol for i in nn_index[:20]]
        for el in elements:
            e = mg.core.Element(el)
            atomic_number = e.Z
            average_ionic_radius = e.average_ionic_radius.real

            # Lowest oxidiation state of the element is used as common oxidation state
            common_oxidation_states = e.common_oxidation_states[0]
            Pauling_electronegativity = e.X
            row = e.row
            group = e.group
            thermal_conductivity = e.thermal_conductivity.real
            boiling_point = e.boiling_point.real
            melting_point = e.melting_point.real
            block = self.block_to_num(e.block)
            IE = e.ionization_energy

        feature_df.loc[len(feature_df)] = [atomic_number, common_oxidation_states, Pauling_electronegativity, 
                                           row, group, thermal_conductivity, boiling_point,
                                           melting_point, block, IE]
        feature = feature_df.to_numpy()
        feature = torch.Tensor(feature).long()
            # feature.unsqueeze_(0).shape
        return feature
