import torch
import torch.nn as nn
from ure.link_predictor import LinkPredictor
import ure.utils as utils


class RelDist(utils.BaseModel):
    def __init__(self, config):
        super().__init__()
        self.encoder = config['encoder_class'](config)
        self.link_predictor = LinkPredictor(config)

    def regularizers(self, pred_relations, mu, std):
        B = pred_relations.shape[0]
        n_rels = pred_relations.shape[1]

        # skewness
        # [B, n_rels]->[B]->1
        loss_s = -(pred_relations * torch.log(pred_relations + 1e-5)).sum(1)

        # dispersion
        # [B, n_rels] -> [n_rels]
        kld_loss = -0.5 * torch.sum(1 + std - mu ** 2 - std.exp(), dim=1)
        kld_loss = torch.mean(pred_relations * kld_loss.unsqueeze(dim=1))
        return loss_s, kld_loss

    def predict_relation(self, _input):
        # [B, n_rels]
        return self.encoder.predict_relation(_input)

    def forward(self, _input):
        # _input contains samples 
        # 'head_ent_samples' [...], 
        # 'tail_ent_samples' [...]
        # [B, n_rels]
        pred_relations, z_out, [mu, var] = self.predict_relation(_input)

        (B, n_rels) = pred_relations.shape
        z_out = torch.cat((pred_relations, z_out), dim=1)
        k = _input['head_ent_samples'].shape[1]
        # link prediction loss
        # [B, n_rels]
        positive_psi, pos_head_bias, pos_tail_bias = self.link_predictor({
            'head_ent': _input['head_ent'],
            'tail_ent': _input['tail_ent']
        }, z_out)
        # [B*k, n_rels]
        negative_psi_head, neg_head_bias, _ = self.link_predictor({
            # [B*k]
            'head_ent': _input['head_ent_samples'].flatten(),
            # B -> [B*k]
            'tail_ent': _input['tail_ent'].unsqueeze(1).repeat(1, k).flatten()
        }, z_out)

        # [B*k, n_rels]
        negative_psi_tail, _, neg_tail_bias = self.link_predictor({
            'head_ent': _input['head_ent'].unsqueeze(1).repeat(1, k).flatten(),
            'tail_ent': _input['tail_ent_samples'].flatten()
        }, z_out)

        # [2B]
        positive_psi = torch.cat(
            [positive_psi + pos_head_bias,
             positive_psi + pos_tail_bias], dim=0)
        # [B*k, n_rels]
        negative_psi_head = negative_psi_head + neg_head_bias
        # [B*k, n_rels]
        negative_psi_tail = negative_psi_tail + neg_tail_bias
        # [2B, n_rels] -> [2B]
        positive_psi = (pred_relations.repeat(2, 1) * positive_psi).sum(dim=-1)
        # [2B]
        positive_psi = nn.functional.logsigmoid(positive_psi)
        # [Bk, n_rels] -> [Bk]
        negative_psi_head = (
            pred_relations.unsqueeze(1)
            * negative_psi_head.view(B, k, n_rels)).sum(dim=-1).flatten()
        negative_psi_head = nn.functional.logsigmoid(-negative_psi_head)
        # [Bk, n_rels] -> [Bk]
        negative_psi_tail = (
            pred_relations.unsqueeze(1)
            * negative_psi_tail.view(B, k, n_rels)).sum(dim=-1).flatten()
        negative_psi_tail = nn.functional.logsigmoid(-negative_psi_tail)

        # [2B] [B*k] [B*k] -> [2B x k+1]
        loss_lp = -torch.cat(
            [positive_psi, negative_psi_head, negative_psi_tail], dim=0)

        loss_s, loss_d = self.regularizers(pred_relations, mu, var)
        loss = torch.cat([
            loss_lp, 
            -0.01 * loss_s, 
            -0.01 * loss_s], dim=0).mean()
        loss_lp = loss_lp.mean()
        loss_s = loss_s.mean()
        loss_details = {
            'lp': loss_lp,
            's': loss_s,
            'd': loss_d
        }
        loss = loss + loss_d
        return loss, loss_details
