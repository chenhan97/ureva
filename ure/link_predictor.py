import torch
import torch.nn as nn
import torch.nn.functional as F
import ure.utils as utils
import math


class LinkPredictor(utils.BaseModel):

    def __init__(self, config):
        super().__init__()
        self.ent_emb = nn.Embedding(config['n_ents'], config['ent_embdim'])
        self.ent_argument_bias = nn.Embedding(config['n_ents'], 1)
        self.ent_embdim = config['ent_embdim']
        self.n_rels = config['n_rels']

        self.init()
        self.r_emb_layers = nn.Sequential(
            nn.Linear(48+config['n_rels'], 47),
            nn.ReLU(),
            nn.Linear(47, self.ent_embdim))
        self.conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(
            2, self.ent_embdim))  
        self.r_mlp = nn.Linear(
            20, config['n_rels'])     
    def init(self):
        self.ent_emb.weight.data.uniform_(-0.01, 0.01)
        self.ent_argument_bias.weight.data.fill_(0.0)

    def forward(self, _input, z):
        # [2Bk] -> [2Bk, D]
        head_emb = self.ent_emb(_input['head_ent'])
        tail_emb = self.ent_emb(_input['tail_ent'])
        # [2Bk, D] bilinear [2Bk, D] -> [2Bk, n_rels]
        r_emb = self.r_emb_layers(z)
        size, r_emb_dim = r_emb.shape[0], r_emb.shape[1]
        if head_emb.shape[0] > r_emb.shape[0]:
            r_emb = torch.repeat_interleave(r_emb, 5, dim=0)
            size = head_emb.shape[0]
        r_emb = r_emb.reshape((head_emb.shape[0], 1, 1, r_emb_dim))
        ent1_emb = head_emb.reshape((head_emb.shape[0], 1, 1, -1))
        ent2_emb = tail_emb.reshape((head_emb.shape[0], 1, 1, -1))
        conv = self.conv(torch.cat((ent1_emb, r_emb, ent2_emb), dim=2))
        score = F.avg_pool2d(conv, kernel_size=(2,1))
        score = self.r_mlp(score.squeeze())
        # [2Bk, 2*D] -> [2Bk, n_rels]

        # [2Bk, n_rels]
        psi = score
        head_bias = self.ent_argument_bias(_input['head_ent'])
        tail_bias = self.ent_argument_bias(_input['tail_ent'])
        return psi, head_bias, tail_bias
