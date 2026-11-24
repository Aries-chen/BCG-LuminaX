# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass, field
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pdb
#from torchgating import TorchGating as TG
#import noisereduce as nr
import random
from scipy.interpolate import interp1d
CUDA_LAUNCH_BLOCKING=1

@dataclass
class YOHOConfig:
    extractor_mode: str = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    processor: str = field(
        default="bcg", metadata={"help": "audio or bcg"}
    )

class YOHOEncoder(nn.Module):
    def __init__(self, 
                extractor_mode: str = "default",
                encoder_embed_dim: int = 768,
                dropout: float = 0.1,
                dropout_input: float = 0.4,
                dropout_features: float = 0.4,
                final_dim: int = 0,
                conv_feature_layers: str = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2",
                feature_len: int = 3,
                anchor_num: int = 3,
                conv_bias: bool = False,
                processor: str = "bcg",
                n_mels: int = 64,
                n_fft: int = 800,
                hop_length: int = 160,
    ):
        super().__init__()
        feature_enc_layers = eval(conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=conv_bias,
        )
        #self attention mechanism
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=encoder_embed_dim,
            num_heads=4,
            dropout=dropout,
        )
        '''
        self.self_attention2 = nn.MultiheadAttention(
            embed_dim=encoder_embed_dim,
            num_heads=4,
            dropout=dropout,
        )
        '''
        if processor == 'audio':
            self.processor = AudioProcessor(target_sr=16000, n_mels=64, n_fft=256, hop_length=160, n_aug_flag=True)
        else:
            self.processor = BCGProcessor(500, 1000, 5)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)
        self.feature_len = feature_len
        self.anchor_num = anchor_num
        final_dim = final_dim if final_dim > 0 else encoder_embed_dim
        #self.conv1d = nn.Conv1d(encoder_embed_dim, final_dim, 1)
        self.final_proj = nn.Linear(encoder_embed_dim, 3*anchor_num)
        self.head_embed = nn.Parameter(torch.randn(1, encoder_embed_dim))
        self.position_embed = nn.Parameter(torch.randn(encoder_embed_dim,10))
        self.head_proj = nn.Linear(encoder_embed_dim, 1)
        self.max_pool = nn.AvgPool1d(feature_len)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: YOHOConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)


    def forward(
        self,
        source,
        #source_a,
        #source_b,
    ):
        source = self.processor.process_audio(source)
        #source = self.processor.process_audio(source)
        #source_a = self.processor.process_audio(source[0])
        #source_b = F.interpolate(source[1].float().unsqueeze(1), scale_factor=source[0].shape[-1]//source[1].shape[-1], mode='linear').squeeze()
        #source_b = self.processor.process_audio(source_b)
        #d = min(source_a.shape[-1], source_b.shape[-1])
        #source = source_a[:,:,:d] + source_b[:,:,:d]
        #source = self.dropout_input(source)
        #source = source[:,:, :-1]
        #energy = energy[:,:, :-1].unfold(2, 100, 100).mean(dim=3)
        features = self.feature_extractor(source).squeeze(-2)#[:,:,1:-1]
        #features = self.feature_extractor(source).reshape(features.shape[0], self.embed, -1)
        #features = (features * energy)#.transpose(1,2).transpose(0,1)
        #pdb.set_trace()
        features += self.position_embed.repeat(features.shape[0],1,1)
        features = torch.concat([features, self.head_embed.repeat(features.shape[0],1).unsqueeze(-1)], dim=2)
        #features += self.position_embed.repeat(features.shape[0],1,1)
        features = features.transpose(1,2).transpose(0,1)
        features, _ = self.self_attention(features, features, features)
        #features = features.transpose(0,1).transpose(1,2)
        #features += self.position_embed.repeat(features.shape[0],1,1)
        #features = features.transpose(1,2).transpose(0,1)
        #features, _ = self.self_attention2(features, features, features)
        features = features.transpose(0,1).transpose(1,2)
        #pdb.set_trace()
        head = features[:,:, -1].squeeze()
        features = features[:,:, :-1]
        cls_result = self.head_proj(head)
        #cls_result = self.head_proj(features.sum(dim=-1))
        #features = features[:,:, :-1].reshape(-1, self.embed*2, 15)
        #features = self.dropout_features(features)
        #pdb.set_trace()
        #seg_cls = self.seg_cls(features.reshape(features.shape[0], -1))
        x = self.final_proj(features.transpose(1,2)) # x.shape = [batch_size, seq_len, 3*anchor_num]
        x = x.reshape([x.shape[0], -1, self.feature_len, x.shape[2]]).transpose(2,3) # x.shape = [batch_size, feature_num, 3*anchor_num, feature_len]
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2], -1]) # x.shape = [batch_size, feature_num, 3*anchor_num*feature_len]
        x = self.max_pool(x).squeeze(-1).reshape([x.shape[0], -1, self.anchor_num * 3]) # x.shape = [batch_size, feature_num, 3*anchor_num]
        result = {
            "x": [x,cls_result]
            #"seg_cls": seg_cls
        }
        return result


    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer
        )
        return res


    def get_losses(self, net_output, sample):
        def conditional_iou_loss(pred_start_end, label_start_end, condition):
            # Compute the intersection and union
            intersection = torch.min(pred_start_end[:, :, 1], label_start_end[:, :, 1]) - torch.max(pred_start_end[:, :, 0], label_start_end[:, :, 0])
            union = torch.max(pred_start_end[:, :, 1], label_start_end[:, :, 1]) - torch.min(pred_start_end[:, :, 0], label_start_end[:, :, 0])

            # Handle cases where intersection is negative (i.e., the two intervals do not overlap)
            intersection = torch.clamp(intersection, min=0)

            # Compute IoU
            iou = intersection / union

            # Use torch.where to conditionally compute the loss
            loss = torch.where(condition, 1 - iou, torch.zeros_like(iou))

            return loss
        label = torch.stack(sample["label"]).to(net_output["x"].device)
        condition = label[:, :, (0,3,6,9,12,15)] == 1
        iou_loss = conditional_iou_loss(net_output["x"][:, :, 1:3], label[:, :, 1:3],condition[:, :, 0])
        iou_loss += conditional_iou_loss(net_output["x"][:, :, 4:6], label[:, :, 4:6],condition[:, :, 1])
        iou_loss += conditional_iou_loss(net_output["x"][:, :, 7:9], label[:, :, 7:9],condition[:, :, 2])
        #再加一个cross_entropy
        binary_cross_entropy = - label[:,:,0] * nn.LogSigmoid()(net_output['x'][:,:,0]) - (1 - label[:,:,0]) * nn.LogSigmoid()(-net_output['x'][:,:,0])
        losses = {}
        losses['iou_loss'] = torch.sum(iou_loss, dim=[-1])
        losses['binary_cross_entropy'] = torch.sum(binary_cross_entropy, dim=[-1])
        return losses

class UNETEncoder(nn.Module):
    def __init__(self,
                 freq: int = 64,
                 sr: int = 8000,
                 kernel_size: int = 3,
                 hop_ratio: int = 100,
                 signal_len: int = 15,
                 init_channels: int = 32,
                 num_stages: int = 5,
                 processor: str = 'audio',
                 norm_dim : List[int] = [1, 2],
                 ):
        super().__init__()
        self.freq = freq
        n_fft = sr
        hop_length = n_fft // hop_ratio
        self.seq_len = (signal_len*sr // hop_length) 
        self.Unet = UNet(
            in_channels=freq,           # 因为Mel谱是单通道的
            out_channels=2,          # 输出一个二分类的结果
            num_stages=num_stages,
            initial_num_channels=init_channels,
            n_freqs=self.freq,
            seq_len=self.seq_len,
            kernel_size=kernel_size,
            pooling='max',
            norm='bn',
            non_lin='leaky_relu' 
        )
        if processor == 'audio':
            self.processor = AudioProcessor(target_sr=sr, n_mels=self.freq, n_fft=n_fft, hop_length=hop_length, n_aug_flag=True)
        else:
            self.processor = BCGProcessor(sr, n_fft, hop_length)
        self.final_proj = nn.Linear(self.seq_len, 1)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: YOHOConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)


    def forward(
        self,
        source,
    ):
        source = self.processor.process_audio(source)
        features = self.Unet(source, classification=False).reshape(-1, 2, self.seq_len).transpose(1,2)
        #seg_cls = self.final_proj(features.transpose(1,2)).squeeze(-1)
        result = {
            "x": features,
            #"seg_cls": seg_cls
        }

        return result

    


    
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv2d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1 # n_mel
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim


    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

class AudioProcessor(nn.Module):
    def __init__(self, target_sr=16000, n_mels=64, n_fft=800, hop_length=160, n_aug_flag=False):
        super().__init__()
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_aug_flag = n_aug_flag
        #self.e_trans = nn.Linear(n_mels, n_mels*4)
        #self.acti = nn.ReLU()
        #self.tg = TG(sr=8000, nonstationary=False, prop_decrease=0.9,n_fft=self.n_fft,hop_length=self.hop_length)
        # 设置重采样和Mel频谱转换

        self.db_transform = T.AmplitudeToDB(stype='power', top_db=80)
        self.aug_time = RandomApplyTime(2, 10,  p=0.8)
        self.aug_spec = RandomApplySpec([T.FrequencyMasking(10),
                                T.TimeMasking(10)],
                                p=0.5)  
              


    def process_audio(self, audio):
        # 转换为张量
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        def calculate_energy(spec, dim=[1]):
            energy = (spec ** 2).mean(dim=dim, keepdim=True).sqrt()  # 在频率和时间维度上求和
            #energy /= (spec.shape[1]*spec.shape[2])  # 标准化能量
            epsilon = 1e-5
            energy += epsilon
            return energy
        def plot_and_save_spec(spec, title = 'mel', filename = 'mel_spec.png'):
            plt.figure(figsize=(10, 4))
            plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower')
            plt.title(title)
            plt.colorbar()
            plt.savefig(filename)
            plt.close()
        def export_audio(audio,index):
            torchaudio.save(str(index)+'_audio.wav', audio[index].unsqueeze(0).cpu(), 8000)
        # 数据增强
        #pdb.set_trace()
        #audio = self.tg(audio) * 10
        audio = torch.where(torch.isnan(audio), torch.zeros_like(audio), audio)
        # 计算Mel频谱
        
        if not self.n_aug_flag:
            #pdb.set_trace()
            if torch.rand(1) < 0.5:
                audio = self.tg(audio) * 10
            audio = self.aug_time(audio)
            mel_spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True, normalized=False, return_complex=False).pow(2).sum(-1)[:,:64,:]
            mel_spec = self.aug_spec(mel_spec)
        else:
            mel_spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True, normalized=False, return_complex=False).pow(2).sum(-1)[:,:64,:]
        #mel_spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True, normalized=False, return_complex=False).pow(2).sum(-1)[:,:64,:]
        #pdb
        #pdb.set_trace()
        mel_energy_freq = calculate_energy(mel_spec, [2])
        mel_spec /= mel_energy_freq #torch.Size([256, 128, 1501])
        #mel_energy_freq = self.e_trans(mel_energy_freq.squeeze(-1))
        #mel_energy_freq = self.acti(mel_energy_freq).unsqueeze(-1)
        #mel_energy = torch.concat([mel_energy_freq, mel_energy_freq], dim=1)
        mel_energy = mel_energy_freq
        return mel_spec, mel_energy
    
class BCGProcessor(nn.Module):
    def __init__(self, target_sr=2000, n_fft=500, hop_length=20, n_aug_flag=False):
        super().__init__()
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_aug_flag = n_aug_flag

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        aug_probs = None
        if aug_probs is None:
            self.aug_probs = {
                'freq_mask': 0.5,
                'time_mask': 0.5,
                'random_crop': 0.5,
                'time_warp': 0.3,
                'amplitude_scaling': 0.3,
                'add_noise': 0.3
            }
        else:
            self.aug_probs = aug_probs
            
            
    def freq_mask(self, spec, mask_param=30):
        transform = T.FrequencyMasking(freq_mask_param=mask_param)
        return transform(spec)

    def time_mask(self, spec, mask_param=30):
        transform = T.TimeMasking(time_mask_param=mask_param)
        return transform(spec)

    def random_crop(self, spec, crop_size):
        time_dim = spec.size(-1)
        freq_dim = spec.size(-2)
        crop_t = time_dim//crop_size
        crop_f = freq_dim//crop_size
        start_t = torch.randint(0, time_dim - crop_t, (1,))
        spec[..., :start_t] = torch.zeros_like(spec[..., :start_t])
        spec[..., start_t+crop_t:] = torch.zeros_like(spec[..., start_t+crop_t:])
        start_f = torch.randint(0, freq_dim - crop_f, (1,))
        spec[:, :start_f,:] = torch.zeros_like(spec[:, :start_f,:])
        spec[:, start_f+crop_f:,:] = torch.zeros_like(spec[:, start_f+crop_f:,:])
        
        return spec

    def time_warp(self, spec):
        # 调整为适合 torchvision 处理的格式：添加一个假的通道维度
        spec = spec.unsqueeze(0)  # [1, Height, Width]
        warped_spec = self.time_warp_transform(spec)
        return warped_spec.squeeze(0)  # 移除通道维度

    def amplitude_scaling(self, spec, scale_factor=0.1):
        scale = 1 + torch.rand(1).item() * scale_factor
        return spec * scale

    def add_noise(self, spec, noise_level=0.05):
        noise = torch.randn_like(spec) * noise_level
        return spec + noise


    def process_audio(self, audio):
        # 转换为张量
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        # 重采样
        # audio = self.resampler(audio)
        audio = torch.where(torch.isnan(audio), audio.mean(dim=-1, keepdim=True), audio)
        audio -= torch.mean(audio, dim=1, keepdim=True)
        # 数据增强
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True, normalized=False, return_complex=False).pow(2).sum(-1)
        #spec = self.spectrogram_transform(audio)
        #spec = spec[:, 2:130, :]
        spec = spec / calculate_energy(spec)
        spec_low = spec[:, :25, :]
        spec_high = spec[:, 25:437, :].reshape(spec.shape[0],103,4,-1).mean(dim=2)
        spec_list = [spec_low, spec_high]
        
        if self.training:
            for s in spec_list:
                if torch.rand(1).item() < self.aug_probs['freq_mask']:
                    s = self.freq_mask(s, mask_param=150)
                if torch.rand(1).item() < self.aug_probs['time_mask']:
                    s = self.time_mask(s, mask_param=300)
                if torch.rand(1).item() < self.aug_probs['random_crop']:
                    s = self.random_crop(s, crop_size=3)  # 假设裁剪大小为100
                if torch.rand(1).item() < self.aug_probs['amplitude_scaling']:
                    s = self.amplitude_scaling(s, scale_factor=2)
                if torch.rand(1).item() < self.aug_probs['add_noise']:
                    s = self.add_noise(s, noise_level=0.05) 

        spec_list[0] = torch.zeros_like(spec_list[0])
        
        spec = torch.cat(spec_list, dim=1)

        return spec
    

class RandomApplyTime(nn.Module):
    def __init__(self, vol = 2 , pitch_shift = 0.5, p=0.8):
        super().__init__()
        self.vol = vol
        self.pitch_shift = pitch_shift
        self.p = p
    
    def magnitude_warping_time_domain(self, waveform, max_warp=0.7):

        """
        Apply magnitude warping directly in the time domain.
        
        Args:
            waveform (torch.Tensor): Input waveform of shape (n_channels, n_samples).
            max_warp (float): Maximum warp factor, determines the intensity of the warping.
            
        Returns:
            torch.Tensor: Magnitude-warped waveform.
        """
        device = waveform.device
        n_samples = waveform.shape[-1]
        # Step 1: Generate a smooth random warp function
        time_steps = torch.linspace(0, n_samples, steps=10, device=device)
        random_factors = torch.empty(10, device=device).uniform_(1 - max_warp, 1 + max_warp)
        interp_func = F.interpolate(torch.arange(n_samples, device=device).float().unsqueeze(0).unsqueeze(0), size=n_samples, mode='linear', align_corners=False).squeeze()
        warp = interp_func.to(waveform.device)
        
        # Step 2: Apply the warp to the waveform
        warped_waveform = waveform * warp
        
        return warped_waveform 


    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            x = T.Vol((torch.rand(1)*self.vol).to(x.device))(x)
            x = self.magnitude_warping_time_domain(x)
            #x = T.PitchShift(8000,torch.ceil(torch.rand(1)*self.pitch_shift)).to(x.device)(x)
        return x


class RandomApplySpec(nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            for transform in self.transforms:
                x = transform(x)
        return x


def calculate_energy(spec, dim=[2]):
    energy = (spec ** 2).mean(dim=dim, keepdim=True).sqrt()  # 在频率和时间维度上求和
    #energy /= (spec.shape[1]*spec.shape[2])  # 标准化能量
    epsilon = 1e-5
    energy += epsilon
    return energy
def plot_and_save_spec(spec, title, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()