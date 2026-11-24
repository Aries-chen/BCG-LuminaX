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
                seq_len: int = 300,
                conv_bias: bool = False,
                processor: dict = {'type':'audio', 'fft':8000, 'hop':160},
                nhead: int = 8,
                num_encoder_layers: int = 6,
                conv_high: dict = {'conv_feature_layers': ' "[(512, 3, 1)] * 3 + [(512, 2, 1)] * 2"','conv_dim':1, 'in_dim':1},
                conv_low: dict = {'conv_feature_layers': ' "[(512, 3, 1)] * 3 + [(512, 2, 1)] * 2"','conv_dim':1, 'in_dim':1},
    ):
        super().__init__()
        self.seq_len = seq_len
        self.final_dim = final_dim
        self.feature_extractor_high = ConvFeatureExtractionModel(
            conv_layers=eval(conv_high['conv_feature_layers']),
            conv_dimension=conv_high['conv_dim'],
            in_d=conv_high['in_dim'],
            mode=extractor_mode,
            conv_bias=conv_bias,
        )
        self.feature_extractor_low = ConvFeatureExtractionModel(
            conv_layers=eval(conv_low['conv_feature_layers']),
            conv_dimension=conv_low['conv_dim'],
            in_d=conv_low['in_dim'],
            mode=extractor_mode,
            conv_bias=conv_bias,
        )
        
        if processor['type'] == 'audio':
            self.processor = AudioProcessor(target_sr=16000, n_mels=64, n_fft=256, hop_length=160, n_aug_flag=True)
        else:
            self.processor = BCGProcessor(processor['fft'], processor['hop'],self.seq_len)
        self.position_embed = nn.Parameter(torch.randn(encoder_embed_dim,seq_len))
        self.head_proj = nn.Linear(encoder_embed_dim, final_dim)
        # 初始化Transformer模型
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_embed_dim, 
                nhead=nhead, 
                dim_feedforward=encoder_embed_dim, 
                dropout=dropout
            ),
            num_layers=num_encoder_layers
        )
        

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
        
        source_down_sampled,source = self.processor.process_audio(source)
        #source_down_sampled = self.processor.process_audio(source.float())
        #source = source[:,:, :-1]
        #energy = energy[:,:, :-1].unfold(2, 100, 100).mean(dim=3)
        features_spec = self.feature_extractor_high(source)[:,:,:-3][:,:,::2]#[:,:,:-2,:].mean(dim=2)
        features_time = self.feature_extractor_low(source_down_sampled)[:,:,:-2]
        #features = self.feature_extractor(source).reshape(features.shape[0], self.embed, -1)
        #features = (features * energy)#.transpose(1,2).transpose(0,1)
        feature = features_time + features_spec
        #feature = torch.cat([features_time, features_spec], dim=1)
        #feature = features_time
        #features = torch.concat([features, self.head_embed.repeat(features.shape[0],1).unsqueeze(-1)], dim=2)
        #feature += self.position_embed.repeat(feature.shape[0],1,1)
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1, self.final_dim).mean(dim=-1)
        feature = feature.transpose(1,2)
        feature = feature.permute(1,0,2)
        feature = self.transformer_encoder(feature)
        feature = feature.permute(1,0,2)
        cls_result = self.head_proj(feature).squeeze(-1)
        cls_result = cls_result.reshape(cls_result.shape[0],-1)
        feature = feature.transpose(1,2)
        result = {
            "x": [cls_result]
            #"seg_cls": seg_cls
        }
        return result


    

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
        conv_dimension: int = 1,
        in_d : int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            conv_dimension,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv(conv_dimension):
                if conv_dimension == 1:
                    conv = nn.Conv1d(n_in, n_out, k, stride=stride, padding=(k ) // 2, bias=conv_bias)
                else:
                    conv = nn.Conv2d(n_in, n_out, k, stride=stride, padding=(k ) // 2, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(conv_dimension),
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
                    make_conv(conv_dimension),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(conv_dimension), nn.Dropout(p=dropout), nn.GELU())

        #in_d = 1 # n_mel
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    conv_dimension,
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
        if len(x.shape) == 2:
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
    def __init__(self,n_fft=500, hop_length=20,seq_len=300):
        super().__init__()
        self.target_sr = 500
        self.seq_len = seq_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
           hop_length=self.hop_length,
        )
        #self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        self.aug_time = RandomApplyTime(2, 10,  p=0.8)

    def process_audio(self, audio):
        # 转换为张量
        '''
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        # 重采样
        # audio = self.resampler(audio)
        audio = torch.where(torch.isnan(audio), audio.mean(dim=-1, keepdim=True), audio)
        '''
        # Create a low-pass filter using a sinc function and Hamming window
        def low_pass_filter(signal, cutoff_freq, sampling_rate, kernel_size=101):
            # Create a time vector centered at zero
            time = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).float()
            
            # Ideal sinc filter (low-pass)
            sinc_filter = torch.sinc(2 * cutoff_freq / sampling_rate * time)
            
            # Apply a Hamming window to the sinc filter
            hamming_window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(kernel_size) / (kernel_size - 1))
            filter_kernel = sinc_filter * hamming_window
            # Normalize filter kernel to ensure unity gain at DC
            filter_kernel /= filter_kernel.sum()
            # Apply the filter using 1D convolution
            filtered_signal = F.conv1d(signal.unsqueeze(1), filter_kernel.unsqueeze(0).unsqueeze(0).to(signal.device), padding=kernel_size // 2)
            
            return filtered_signal.squeeze()
        
        # Decimation: Take every nth sample after filtering
        def decimate(signal, factor):
            return signal[:,::factor]
        
        def calculate_energy(spec, dim=[2]):
            energy = (spec ** 2).mean(dim=dim, keepdim=True).sqrt()  # 在频率和时间维度上求和
            #energy /= (spec.shape[1]*spec.shape[2])  # 标准化能量
            epsilon = 1e-5
            energy += epsilon
            return energy
        def plot_and_save_spec(spec, title = 'spec', filename = 'spec.png'):
            plt.figure(figsize=(10, 4))
            plt.imshow(spec.cpu().numpy(), aspect='auto', origin='lower')
            plt.title(title)
            plt.colorbar()
            plt.savefig(filename)
            plt.close()
        def plot_and_save_plot(audio_resampled, title = 'signal', filename = 'signal.png'):
            plt.figure()
            plt.plot(audio_resampled.cpu().numpy())
            plt.title(title)
            plt.savefig(filename)
            plt.close()
        
        
        audio_resampled = torch.empty((audio['bcg_raw'].shape[0],0,self.seq_len), dtype=torch.float32).to(audio['bcg_raw'].device)  
        
        for k, v in audio.items():
            v = torch.tensor(v, dtype=torch.float32)
            mask = audio['body_movement'].bool()
            if k == 'bcg_raw':
                spec = self.spectrogram_transform(v)
                spec /= calculate_energy(spec)
                mask = mask[:,::2]
                v = low_pass_filter(v, 1, self.target_sr, kernel_size=101)
                v = decimate(v, 250)
                v_mean = v.masked_fill(mask, 0).sum(dim=-1, keepdim=True) / (~mask).sum(dim=-1, keepdim=True)
                v_std = ((v.masked_fill(mask, 0) - v_mean) ** 2).sum(dim=-1, keepdim=True) / (~mask).sum(dim=-1, keepdim=True)
                v_std = torch.sqrt(v_std) + 1e-5

                v = (v - v_mean) / v_std
                # 用均值填充 mask 元素
                v_mean_expanded = v_mean.expand_as(v)
                v[mask] = v_mean_expanded[mask]
                audio_resampled = torch.cat([audio_resampled, v.reshape(v.shape[0],self.seq_len,-1).permute(0,2,1)], dim = 1)
            elif k != 'body_movement':
                #以bodymovement作为mask，为1的mask掉，填充为均值
                v_mean = v.masked_fill(mask, 0).sum(dim=-1, keepdim=True) / (~mask).sum(dim=-1, keepdim=True)
                v_std = ((v.masked_fill(mask, 0) - v_mean) ** 2).sum(dim=-1, keepdim=True) / (~mask).sum(dim=-1, keepdim=True)
                v_std = torch.sqrt(v_std) + 1e-5

                v = (v - v_mean) / v_std
                # 用均值填充 mask 元素
                v_mean_expanded = v_mean.expand_as(v)
                v[mask] = v_mean_expanded[mask]
                audio_resampled = torch.cat([audio_resampled, v.reshape(v.shape[0],self.seq_len,-1).permute(0,2,1)], dim = 1)
            
        return audio_resampled,spec
    

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


