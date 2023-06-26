import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
    Referenced from Github: https://github.com/huggingface/transformers
"""


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # (1, n_position, d_hid)

    def forward(self, x):
        y = self.pos_table[:, :x.size(1)].clone().detach()
        return x + y


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, 1, d_k)
        k = self.w_ks(k).view(sz_b, len_k, 1, d_k)
        v = self.w_vs(v).view(sz_b, len_v, 1, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class Encoder(nn.Module):
    def __init__(
            self, n_src_vocab, d_model, n_position, d_k=164, d_v=164,
            pad_idx=0, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attention = SelfAttention(d_model, d_k, d_v, dropout=dropout)

    def forward(self, src_seq, mask):
        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        enc_output, enc_slf_attn = self.self_attention(enc_output, enc_output, enc_output, mask=mask)

        return enc_output
