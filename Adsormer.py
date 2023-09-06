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


import csv
import pandas as pd

# from data import CIFData, AtomCustomJSONInitializer, GaussianDistance
import os
import csv
import random



def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate , head_size = 8):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        
        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias = False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias = False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias = False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)
        
        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias = False)
        initialize_weight(self.output_layer)
        
    def forward(self, q, k, v):
        orig_q_size = q.size()
        
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        
        # head_i 
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1 , self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        
        q= q.transpose(1,2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2,3)
        
        # scaled dot product
        q.mul_(self.scale)
        x = torch.matmul(q,k)
        x = torch.softmax(x, dim = 3)
        x = self.att_dropout(x)
        x= x.matmul(v)
        
        x= x.transpose(1,2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        x = self.output_layer(x)
        return (x)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.BatchNorm1d(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.BatchNorm1d(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x.transpose(1,2)).transpose(1,2)
        y = self.self_attention(y, y, y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x.transpose(1,2)).transpose(1,2)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

# class DecoderLayer(nn.Module):
#     def __init__(self, hidden_size, filter_size, dropout_rate):
#         super(DecoderLayer, self).__init__()
        
#         self.self_attention_norm = nn.LayerNorm(hidden_size, eps = 1e-6)
#         self.self_attention = MultiHeadAttentioin(hidden_size=hidden_size, dropout_rate=dropout_rate)
#         self.self_attention_dropout = nn.Dropout(dropout_rate)
        
#         self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.enc_dec_attention = MultiHeadAttentioin(hidden_size, dropout_rate)
#         self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)
        
#         self.ffn_norm =  nn.LayerNorm(hidden_size, eps = 1e-6)
#         self.ffn =  FeedForwardNetwork(hidden_size, filter_size,  dropout_rate)
#         self.ffn_dropout = nn.Dropout(dropout_rate)
        
#     def forward(self, x, enc_output):
#         y = self.self_attention_norm(x)
#         y = self.self_attention(y,y,y)
#         y = self.self_attention_dropout(y)
#         x = x+y
        
#         y = self.enc_dec_attention_norm(x)
#         y = self.enc_dec_attention_norm(y, enc_output, enc_output)
#         y = self.enc_dec_attention_dropout(y)
#         x= x+y
        
#         y= self.ffn_norm(x)
#         y = self.ffn(y)
#         y = self.ffn_dropout(y)
#         x = x+y
#         return x
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y)
        y = self.self_attention_dropout(x)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Encoder, self).__init__()
        
        encoders = [EncoderLayer(hidden_size= hidden_size, filter_size= filter_size, dropout_rate = dropout_rate)
                   for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.hidden_size = hidden_size
        
        self.last_norm = nn.BatchNorm1d(hidden_size, eps = 1e-6)
        
    def forward(self, inputs):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output)
        return self.last_norm(encoder_output.transpose(1,2)).transpose(1,2)
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()
        
        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                   for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)
        self.last_norm = nn.LayerNorm(hidden_size, eps = 1e-6 )
        
    def forward(self, targets, enc_output):
        decoder_ouput = targets
        for dec_layer in self.layers:
            decoder_ouput= dec_layer(decoder_ouput, enc_output)
        return self.last_norm(decoder_ouput)
    

    
    
class Transformer(nn.Module):
    def __init__(self, nn_nums, feature_nums, n_layers = 6, hidden_size = 512,
                filter_size = 2048, dropout_rate = 0.1):
        super(Transformer, self).__init__()
        
        self.hidden_size = hidden_size
        self.emb_scale=  hidden_size ** 0.5
        
        # self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        self.input_normalize = nn.BatchNorm1d(feature_nums,eps =1e-6)
        self.target_normalize = nn.LayerNorm(1, eps= 1e-6)
        
        # self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        # nn.init.normal_(self.t_vocab_embedding.weight, mean = 0, std = hidden_size ** -0.5)
        # self.t_emb_dropout = nn.Dropout(dropout_rate)
        
        self.i_vocab_embedding1 = nn.Linear(feature_nums,hidden_size)
        # self.i_vocab_embedding2 = nn.Linear(1, hidden_size)
        nn.init.normal_(self.i_vocab_embedding1.weight, mean = 0 , std = hidden_size ** -0.5)
        self.i_emb_dropout = nn.Dropout(dropout_rate)
        self.encoder = Encoder(hidden_size, filter_size, dropout_rate, n_layers)
        self.decoder = Decoder(hidden_size, filter_size, dropout_rate, n_layers)
        # self.out = nn.Linear(t_vocab_size *1, 1)
        
        self.out1 = nn.Linear(hidden_size, 1)
        self.out2= nn.Linear(nn_nums,1)
        
    
    def forward(self, inputs, targets):
        # input_normed = self.input_normalize(inputs.float()).long()
        batch_size = inputs.size(0)
        enc_output = self.encode(inputs)
        out1= self.out1(enc_output).squeeze()
        out2= self.out2(out1).squeeze()
        return  out2
    
    def encode(self, inputs):
        inputs = self.input_normalize(inputs.transpose(1,2)).transpose(1,2)
        # Input embedding
        input_embedded = self.i_vocab_embedding1(inputs)
        # input_embedded = self.i_vocab_embedding2(input_embedded)
        input_embedded *- self.emb_scale
        input_embedded = self.i_emb_dropout(input_embedded)
        
        return self.encoder(input_embedded)
    
    
    
#     def decode(self, targets, enc_output):
#         # target embedding
#         targets = self.target_normalize(targets.view(-1,1).to(torch.float32)).long()
#         target_embedded = self.t_vocab_embedding(targets.long())
#         # target_embedded *= self.emb_scale
#         target_embedded = self.t_emb_dropout(target_embedded).to(torch.float32)
#         decoder_output = self.decoder(target_embedded, enc_output)
#         # output = torch.matmul(decoder_output, self.t_vocab_embedding.weight.transpose(0,1))
#         weights =self.t_vocab_embedding.weight.unsqueeze(0).transpose(1,2)
#         # output = torch.matmul(decoder_output, weights)
#         # output = self.out(output).squeeze()
#         output = self.out(decoder_output.squeeze()).view(-1)

#         return output
    
        
        
