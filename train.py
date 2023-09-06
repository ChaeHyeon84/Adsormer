import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read,write
from ase import Atoms, Atom
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
from torch.utils.data.dataloader import default_collate

import csv
import pandas as pd

# from data import CIFData, AtomCustomJSONInitializer, GaussianDistance
import os
import csv
import random
import argparse
import time

def remove_adsorbates(atoms):
    binding_sites = []
    adsorbates_index = []
    
    for atom in atoms:
        if atom.tag == 1:
            binding_sites.append([atom.symbol,list(atom.position)])
            adsorbates_index.append(atom.index)
    del atoms[adsorbates_index]
    bare_slab = atoms.copy()
    
    return binding_sites, bare_slab

def get_nearest_atoms(atoms):
    # view(atoms)
    binding_sites, slab = remove_adsorbates(atoms)
    binding_sites.sort(key = lambda x: x[1][2] )    
    
    copied_atom = slab.copy()
    copied_atom += Atom(binding_sites[0][0],binding_sites[0][1],tag =1 )
    copied_atom = copied_atom.repeat((3,3,1))
    ads_index = np.where((copied_atom.get_tags()) ==1)[0][4]
    
    structure = AseAtomsAdaptor.get_structure(copied_atom)
    # nn = structure.get_neighbors(site=structure[ads_index] , r= min(structure.lattice.abc))
    nn = structure.get_neighbors(site=structure[ads_index] , r= 10)
    nn.sort(key = lambda x : x[1]) # sort nearest atoms
    nn_index = [nn[i][2] for i in range(len(nn))]
    nn_distances =[nn[i][1] for i in range(len(nn))]
    
    
    # view(copied_atom)
    return copied_atom, nn_index,nn_distances

def get_atom_property(atoms):
    global feature
    feature_df = pd.DataFrame(columns=[f'feat{i}' for i in range(10)])
    atom, nn_index,nn_distances = get_nearest_atoms(atoms)
    elements = [atom[i].symbol for i in nn_index[:15]]
    distances = nn_distances[:15]
    def block_to_num(block):
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

    for el, distance in zip(elements, distances):
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
        block = block_to_num(e.block)
        IE = e.ionization_energy
        
        feature_df.loc[len(feature_df)] = [atomic_number, common_oxidation_states, Pauling_electronegativity, 
                                           row, group, thermal_conductivity, boiling_point,
                                           melting_point, block, IE]
    feature = feature_df.to_numpy()
    feature = feature/distance
    feature = torch.Tensor(feature)
        # feature.unsqueeze_(0).shape
    return feature

def train(train_loader, model,global_step, criterion, optimizer, epoch):
    train_mae_epoch = []
    model.train()
    
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
        # mae_erros = []
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        if torch.cuda.is_available():
            inputs= inputs.to('cuda')
            targets = targets.to('cuda')
        # print(inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs,targets)
        outputs =  outputs.type(torch.float32)
        loss = criterion(outputs, targets.to(torch.float32))
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        # print(f'[{epoch +1}, {i+1:5d}] loss: {loss:.4f}')
        mae_error = torch.mean(abs(outputs-targets))
        train_mae_epoch.append(mae_error)
        # print(f'mae error: {mae_error} ')
        if i % 10 == 9:
            # print(f'[{epoch +1}, {i+1:5d}] loss: {running_loss/ 2000:.3f}')
            # print(f'mae error: {mae_error} ')
            running_loss = 0.0
        global_step += 1
    mae_avg = torch.mean(torch.Tensor(train_mae_epoch))
    print(f'[{epoch +1}, {i+1:5d}] Train loss: {loss:.4f}  Train MAE: {mae_avg:.3f}')
    return global_step

def validation(val_loader , model, global_step,criterion):
    mae_epoch_val = []
    test_targets = []
    test_preds = []
    # test_cif_ids = []
    model.eval()
    total_loss = 0.0
    total_cnt = 0
    for i, data in enumerate(val_loader,0):
        # mae_erros = []
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        a,b = inputs.shape, targets.shape
        if torch.cuda.is_available():
            inputs= inputs.to('cuda')
            targets = targets.to('cuda')
            
        pred = model(inputs, targets)
        pred =  pred.type(torch.float32)
        loss = criterion(pred, targets.to(torch.float32))
        mae = torch.mean(abs(pred-targets))
        mae_epoch_val.append(mae)
        total_loss += loss
        total_cnt += len(inputs)
        test_preds += pred.view(-1).tolist()
        test_targets += targets.view(-1).tolist()
    
    val_loss = total_loss / total_cnt
    mae_avg = torch.mean(torch.Tensor(mae_epoch_val))
    print(f"Validation Loss: {val_loss:.4f}   MAE: {mae_avg:.3f}")
    import csv
    with open('test_results.csv', 'w') as f:
        writer = csv.writer(f)
        for target, pred in zip(test_targets,
                                        test_preds):
            writer.writerow((target, pred))
        
    return val_loss

# def collate_fn(batch):
#     # TODO: Implement your function
#     # But I guess in your case it should be:
#     return tuple(batch)

class make_dataset(Dataset):
    def __init__(self, root_dir, dmin = 0, step = 0.2, random_seed = 123):
        self.root_dir = root_dir
        target_file = os.path.join(self.root_dir, 'data/co_target.csv')
        target_df = pd.read_csv(f'{self.root_dir}data/co_target.csv')[:1000]
        random.seed(random_seed)
        self.target_df = target_df.sample(frac=1).reset_index(drop= True)
    
    def __len__(self):
        return len(self.target_df)
    
    def __getitem__(self, idx):
        traj_id = self.target_df['name'][idx]
        target = self.target_df['target'][idx]
        atoms = read(f'{self.root_dir}final_CO_slab/{traj_id}.traj')
        feature_data = get_atom_property(atoms)
        
        
        return feature_data, target
    
def collate_fn(dataset_list):
    """
    list of tuples for each data point.
    """
    batch_feature = []
    batch_target = []
    for  i, (feature, target) in enumerate(dataset_list):
        batch_feature.append(feature)
        batch_target.append(target)
    batch_feature = torch.nn.utils.rnn.pad_sequence(batch_feature, batch_first = True) 
    batch_target = torch.Tensor(batch_target)
    return batch_feature,batch_target
    
def get_train_val_test_loader(dataset, train_ratio=0.7, batch_size=50 ,valid_ratio=0.15, test_ratio=0.15, num_workers=1, collate_fn = default_collate,pin_memory=False):
    
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
    train_loader = DataLoader(dataset, batch_size = batch_size,sampler=train_sampler, num_workers= num_workers, collate_fn= collate_fn,pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers ,collate_fn= collate_fn,pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size= batch_size, sampler = test_sampler, num_workers=num_workers,collate_fn= collate_fn,pin_memory=pin_memory )

    return train_loader, val_loader, test_loader




      
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_nums',type = int,  default= 15)
    parser.add_argument('--train_step', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--feature_nums', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--filter_size', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=16000)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='Adsormer')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--no_cuda', action='store_true')
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=0.7, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.15, type=float, metavar='N',
                        help='percentage of validation data to be loaded (default '
                             '0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                             help='number of validation data to be loaded (default '
                                  '1000)')
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.15, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                            help='number of test data to be loaded (default 1000)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--parallel', action='store_true')
    # parser.add_argument('--summary_grad', action='store_true')
    opt = parser.parse_args()
    global_step = 0.
    
    dataset= make_dataset(root_dir= '/home/cut6089/research/GASpy/')
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, batch_size= opt.batch_size,
                                                                        train_ratio = opt.train_ratio, valid_ratio= opt.val_ratio,
                                                                        test_ratio = opt.test_ratio,pin_memory = 'cuda', collate_fn= collate_fn)
   
    if opt.parallel and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # device = torch.device('cpu' if opt.no_cuda else 'cuda')
    
    
    from Adsormer import Transformer
    model = Transformer(nn_nums = opt.nn_nums, feature_nums = opt.feature_nums,n_layers=opt.n_layers, hidden_size=opt.hidden_size,
                         filter_size=opt.filter_size, dropout_rate=opt.dropout)
    
    model = model.to('cuda')
    criterion = nn.MSELoss()
    if opt.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = 0.001,
                              momentum= 0.9)
    elif opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in  range(opt.train_step):
        print("Epoch", epoch)
        train(train_loader, model,global_step=global_step, criterion=criterion, optimizer= optimizer, epoch=epoch)
        val_loss = validation(val_loader,model= model, global_step=global_step ,criterion=criterion)

if __name__ == '__main__':
    main()

    
