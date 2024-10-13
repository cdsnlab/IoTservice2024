from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs, args):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift)

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned_2, dim_feedforward=2*configs.TSlength_aligned_2,nhead=1)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned_2 * configs.input_channels_2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )                   
        self.shift_cls_layer_f = nn.Linear(configs.TSlength_aligned_2 * configs.input_channels_2, args.K_shift_f)

        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, configs.num_classes)    


    def forward(self, x_in_t, x_in_f):

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Shifted transformation classifier"""
        s_time = self.shift_cls_layer_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f.float())
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        """Shifted transformation classifier"""
        s_freq = self.shift_cls_layer_f(h_freq)

        return h_time, z_time, s_time, h_freq, z_freq, s_freq




"""Two contrastive encoders"""
class TFC_one(nn.Module):
    def __init__(self, configs, args):
        super(TFC_one, self).__init__()
        self.training_mode = args.training_mode

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, batch_first = True)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, 1)

        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, configs.num_classes)     
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_t):

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Shifted transformation classifier"""
        s_time = self.sigmoid(self.shift_cls_layer_t(h_time))


        return h_time, z_time, s_time

class TFC_class(nn.Module):
    def __init__(self, configs, args, class_len):
        super(TFC_class, self).__init__()
        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, class_len)     


    def forward(self, x_in_t):

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)

        """Shifted transformation classifier"""
        s_time = self.linear(h_time)

        return h_time, s_time



"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        #self.logits_simple = nn.Linear(64, configs.num_classes)
        self.logits_simple = nn.Linear(64, 2)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred

"""One contrastive encoders"""
class TFC_GOAD(nn.Module):
    def __init__(self, configs, num_classes):
        super(TFC_GOAD, self).__init__()

        encoder_layers = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.projector = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.linear = nn.Linear(256, num_classes)    


    def forward(self, x_in):

        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z = self.projector(h)

        """Shifted transformation classifier"""
        s = self.linear(z)

        return h, z, s