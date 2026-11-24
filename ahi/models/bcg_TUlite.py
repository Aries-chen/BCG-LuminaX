import torch
import torch.nn as nn
import pdb
import torchvision
import sys
import time
if sys.platform != 'win32':
    import readline
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import torchaudio

import torchaudio.transforms as T
import torch.nn.functional as F

class TUlite(nn.Module):
    def __init__(self):
        super().__init__()

        channel_first = 1
        channel_sec = 8
        channel_trd = 16
        channel_fur = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_first, out_channels=channel_sec, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_first),  # 
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_sec, out_channels=channel_sec, kernel_size=(1,1), padding=(0,0), stride=1),  # 
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_sec, out_channels=channel_sec, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_sec),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_sec, out_channels=channel_sec, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_sec),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_sec, out_channels=channel_trd, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_sec),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_trd, out_channels=channel_trd, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_trd, out_channels=channel_trd, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_trd),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_trd, out_channels=channel_trd, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_trd),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)  # 152
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_trd, out_channels=channel_fur, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_trd),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(15,15), padding=(7,7), stride=1, groups=channel_fur),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_fur),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )

        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=8)

        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(15,15), padding=(7,7), stride=1,
                      groups=channel_fur),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(15,15), padding=(7,7), stride=1,
                      groups=channel_fur),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_fur, out_channels=channel_fur, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_fur),

            nn.ConvTranspose2d(channel_fur, channel_trd, kernel_size=(2,2), stride=2)
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_fur, out_channels=channel_trd, kernel_size=(15,15), padding=(7,7), stride=1),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_trd, out_channels=channel_trd, kernel_size=(15,15), padding=(7,7), stride=1,
                      groups=channel_trd),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_trd, out_channels=channel_trd, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_trd),

            nn.ConvTranspose2d(channel_trd, channel_sec, kernel_size=(2,2), stride=2)
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_trd, out_channels=channel_sec, kernel_size=(15,15), padding=(7,7), stride=1),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_sec, out_channels=channel_sec, kernel_size=(15,15), padding=(7,7), stride=1,
                      groups=channel_sec),  # 300
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_sec, out_channels=channel_sec, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),

            nn.BatchNorm2d(channel_sec),


        )


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.halfer1 = self.half_pool(channel_first, channel_trd)
        self.halfer2 = self.half_pool(channel_trd, channel_fur)
        self.halfer6 = self.half_pool(channel_sec, channel_sec)

        self.channel_attention_c1 = self.channel_attention(channel_sec,4)
        self.channel_attention_c2 = self.channel_attention(channel_trd, 4)

        self.dimprocess = self.dimp()
        self.indimprocess = self.indimp() 



        self.fla = nn.Sequential(
            nn.Conv2d(in_channels=channel_sec, out_channels=channel_first, kernel_size=(1,1), padding=(0,0), stride=1),  # 300
            nn.PReLU(),
            nn.Flatten(),
        )
       

        self.fc = nn.Sequential(         
            nn.Linear(19200, 3000, bias=False),
            nn.PReLU(),
            nn.Linear(3000, 300, bias=False),
            nn.PReLU(),
            #nn.Softmax(dim=1)

        )
        

    def half_pool(self, channel_input, channel_output):
        return nn.Sequential(
            nn.Conv2d(in_channels=channel_input, out_channels=channel_output, kernel_size=1,stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def channel_attention(self, channel_input,reduction):
        return nn.Sequential(
            nn.Conv2d(channel_input,channel_input // reduction, 1, bias=False), #[2, 16, 1]
            nn.ReLU(inplace=True), #[2, 1, 1]
            nn.Conv2d(channel_input// reduction,channel_input, 1, bias=False), #[2, 16, 1]
            nn.Sigmoid(),
        )

    def dimp(self):
        return nn.Sequential(
           nn.Conv1d(2048, 512, kernel_size=1, stride=1)
        )  

    def indimp(self):
        return nn.Sequential(
            nn.Conv1d(512, 2048, kernel_size=1, stride=1)
        )  


    def fre_reshape(self,signal):
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
            #pdb.set_trace()
            filtered_signal = F.conv1d(signal.unsqueeze(1), filter_kernel.unsqueeze(0).unsqueeze(0).to(signal.device), padding=kernel_size // 2)
            
            return filtered_signal.squeeze()
        
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

        spectrogram_transform = T.Spectrogram(n_fft=508, hop_length=500).to(signal.device)
        spec = spectrogram_transform(signal.unsqueeze(1))
        bs = spec.shape[0]
        spec /= calculate_energy(spec)
        pec = spec[:,:,1:,1:]

        signal_lpass = low_pass_filter(signal.squeeze(),1,500).unsqueeze(1)
        
        #fft_sig = torch.fft.fft(signal)

        
        resampled_signal = F.interpolate(signal_lpass, size=(600,), mode='nearest')
        mean_v = torch.mean(resampled_signal)
        std_v = torch.std(resampled_signal)
        std_rs_signal = ((resampled_signal - mean_v) / std_v).reshape(bs,1,2,300)
        #print(pec.shape, resampled_signal.shape)
        res_con = torch.cat((pec,std_rs_signal),dim=2)#.permute(0,1,3,2)
          
        
        plot_and_save_spec(pec[0,0,:,:])
        #plot_and_save_plot(signal[0,0,0,:])   
        #pdb.set_trace()
        return bs, res_con         
             
    def forward(self, x):

        x = x['bcg_raw'].float()
        #pdb.set_trace()

        bs, x = self.fre_reshape(x)
        half_x1 = self.halfer1(x)

        process1 = self.conv1(x)
        res1 = x + process1

        process2 = self.conv2(res1)
        res2 = half_x1 + process2

        half_x2 = self.halfer2(res2)
        
        process3 = self.conv3(res2)
        res3 = half_x2 + process3
        
        trans_input =self.dimprocess(res3.reshape(bs,2048,75)).permute(2,0,1)
        tranfm = self.transformer_encoder(trans_input).permute(1,2,0)
        aft_tranfm = self.indimprocess(tranfm).reshape(bs,32,64,75)

        channel_max1 = self.max_pool(res1)
        channel_max2 = self.max_pool(res2)

        ch_attention_max1 = self.channel_attention_c1(channel_max1)
        ch_attention_max2 = self.channel_attention_c2(channel_max2)
        out_attmax1 = res1 * ch_attention_max1
        out_attmax2 = res2 * ch_attention_max2

        process4 = self.deconv1(aft_tranfm)
        #print(process4.shape,out_attmax2.shape)
        process5 = self.deconv2(torch.cat((process4,out_attmax2),dim=1))
        process6 = self.deconv3(torch.cat((process5,out_attmax1),dim=1))

        aft_ml = self.halfer6(process6)

        final = self.fla(aft_ml)

        out = self.fc(final)
        out = torch.squeeze(out)
        #pdb.set_trace()
        return out

def main():
    device = torch.device("cuda:2")  
    net = TUlite().to(device)
    
    
    input = torch.randn(15,150000).to(device)
    output = net(input)
    print(output,output.shape)

  
    
if __name__ == '__main__':
    main()