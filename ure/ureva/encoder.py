import torch
import torch.nn as nn
import ure.utils as utils
from ure.dataset import MAX_POS
import math
import sys
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class Encoder(utils.BaseModel):
    def __init__(self, config):
        super().__init__()

        # word and position embeddings
        self.word_embs = nn.Embedding(config['n_words'], config['word_embdim'])
        self.word_embs.weight.data.uniform_(-0.001, 0.001)
        self.etype_embs = nn.Embedding(config['n_etype_with_subjobj'], 120)
        self.bias = nn.Parameter(torch.Tensor(120))
        input_dim = config['word_embdim']
        self.etype_embs.weight.data.uniform_(-0.001, 0.001)
        self.bias.data.fill_(0.0)
        self.sen_dim = 120
        self.classifier = nn.Linear(self.sen_dim, config['n_rels'])
        self.z_enco = nn.Linear(self.sen_dim, 48)
        self.z_mlp = nn.Linear(config['n_rels']+48, 48)
        self.mu_mlp = nn.Linear(48, 48)
        self.var_mlp = nn.Linear(48, 48)
    def encode_sentence(self, _input):
        head = self.etype_embs(_input['head_etype'])
        tail = self.etype_embs(_input['tail_etype'])
        # [B, 2D]
        output = torch.cat([head, tail], dim=1)
        # This is equivalent to FFNN: [head,tail]W + b
        sent_embs = head + tail + self.bias
        
        return sent_embs

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mu)

    def forward(self, _input):
        sent_embs = self.encode_sentence(_input)
        logits = self.classifier(sent_embs)
        sent_embs = self.encode_sentence(_input)
        z_encoding = self.z_enco(sent_embs)
        z_out = self.z_mlp(torch.cat((z_encoding, logits), dim=1))
        mu = self.mu_mlp(z_out)
        var = self.var_mlp(z_out)
        sample_z = self.reparametrize(mu, var)
        return logits, sample_z, [mu, var]

    def predict_relation(self, _input):
        logits, sample_z, [mu, var] = self.forward(_input)
        return nn.functional.softmax(logits, dim=1), sample_z, [mu, var]

