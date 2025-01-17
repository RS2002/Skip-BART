from torch.ao.nn.quantized import Sigmoid
from transformers import BartModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0):
        super().__init__()
        self.arl = arl
        self.attention = nn.Sequential(
            nn.Linear(layer_sizes[0],layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes[0],layer_sizes[0])
        )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x


class BART(nn.Module):
    def __init__(self,bartconfig, class_num = 100):
        super().__init__()
        d_model = bartconfig.d_model
        self.decoder_emb = nn.Embedding(class_num,d_model)
        self.bart = BartModel(bartconfig)

    def forward(self, x_encoder, x_decoder, attn_mask_encoder = None, attn_mask_decoder = None):
        emb_encoder = x_encoder
        emb_decoder = self.decoder_emb(x_decoder)
        y = self.bart(inputs_embeds=emb_encoder, decoder_inputs_embeds=emb_decoder,
                      attention_mask=attn_mask_encoder, decoder_attention_mask=attn_mask_decoder,
                      output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def encode(self, x_encoder, attn_mask_encoder = None):
        emb_encoder = x_encoder
        y = self.bart.encoder(inputs_embeds=emb_encoder, attention_mask=attn_mask_encoder, output_hidden_states=False)
        y = y.last_hidden_state
        return y


class ML_BART(nn.Module):
    def __init__(self, bartconfig, class_num = [180,256], pretrain = False):
        super().__init__()
        d_model = bartconfig.d_model
        self.decoder_emb = nn.ModuleList([
            nn.Embedding(class_num[0] + 1, d_model),
            nn.Embedding(class_num[1] + 1, d_model)
        ])
        self.decoder_fusion = MLP([d_model*2,d_model])
        self.bart = BartModel(bartconfig)
        self.pretrain = pretrain

    def forward(self, x_encoder, x_decoder, attn_mask_encoder = None, attn_mask_decoder = None):
        emb_encoder = x_encoder

        if self.pretrain:
            emb_decoder = x_decoder
        else:
            emb_decoder = torch.concatenate([self.decoder_emb[0](x_decoder[...,0]),self.decoder_emb[1](x_decoder[...,1])],dim=-1)
            emb_decoder = self.decoder_fusion(emb_decoder)

        y = self.bart(inputs_embeds=emb_encoder, decoder_inputs_embeds=emb_decoder,
                      attention_mask=attn_mask_encoder, decoder_attention_mask=attn_mask_decoder,
                      output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def encode(self, x_encoder, attn_mask_encoder = None):
        emb_encoder = x_encoder
        y = self.bart.encoder(inputs_embeds=emb_encoder, attention_mask=attn_mask_encoder, output_hidden_states=False)
        y = y.last_hidden_state
        return y

    def reset_decoder(self):
        for name, param in self.bart.decoder.named_parameters():
            if param.dim() >= 2:
                init.xavier_uniform_(param)
            elif param.dim() == 1:
                init.zeros_(param)


class ML_Classifier(nn.Module):
    def __init__(self, hidden_dim = 512, class_num = [180,256]):
        super().__init__()
        self.classifier = nn.ModuleList([
            MLP([hidden_dim,hidden_dim,class_num[0] + 1]),
            MLP([hidden_dim, hidden_dim, class_num[1] + 1])
        ])

    def forward(self, x):
        h = self.classifier[0](x)
        v = self.classifier[1](x)
        return h,v


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class Sequence_Classifier(nn.Module):
    def __init__(self, class_num=1, hs=512, da=512, r=8):
        super().__init__()
        self.attention = SelfAttention(hs, da, r)
        self.classifier = MLP([hs * r, (hs * r + class_num)// 2, class_num])

    def forward(self, x):
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


class Token_Predictor(nn.Module):
    def __init__(self, hidden_dim=512, class_num=1):
        super().__init__()
        self.classifier = MLP([hidden_dim, (hidden_dim+class_num)//2, class_num])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x