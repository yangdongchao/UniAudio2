try:
    from .rvq import *
except:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rvq import *

try:
    from ..modules.random_quantizer import RandomProjectionQuantizer
    from ..modules.features import MelSTFT
    from ..modules.conv import Conv2dSubsampling
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.random_quantizer import RandomProjectionQuantizer
    from modules.features import MelSTFT
    from modules.conv import Conv2dSubsampling


class RVQDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        normalize: bool = False,
    ):
        self.sample_rate = sample_rate
        self.datas,inds,tot,self.sizes = load_audio_by_json(manifest_path, None, None, self.sample_rate)
        self.dataset_len = len(self.datas)

        self.reader = Read_and_PadCrop_Normalized_T(n_samples=CLIPSECS*sample_rate,sample_rate = self.sample_rate)
        self.normalize = normalize
    

    def __getitem__(self, i):
        index = i
        item = None
        while item is None:
            try:
                wav = self.get_audio_by_slice(index)
                item = {"id": index, "source": wav}
            except Exception as e:
                # print(e)
                traceback.print_exc()
                print(f'skip damaged data {index}')
                index = np.random.randint(0,len(self.sizes)-1)
        return item

    def __len__(self):
        return self.dataset_len
    
    def get_audio_by_slice(self,index):
        
        
        wav_path = self.datas[index]['path']
    
        audio_info =  torchaudio.info(wav_path)
        origin_sample_rate = audio_info.sample_rate
        origin_duration = audio_info.num_frames / origin_sample_rate

        wav, *ignored = self.reader(wav_path, origin_duration,origin_sample_rate)
        wav = wav.float()
        
        wav = wav.permute(1,0)
        wav = self.postprocess(wav, self.sample_rate) 
        return wav
    
    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

class Preprocessor(nn.Module):
    def __init__(self, 
            codebook_dim=16,
            codebook_size=4096,
            hop_length=240,
            n_mels=128,
            stat_path='our-MERT/data/musicfm/msd_stats.json',
        ) -> None:
        super().__init__()

        self.features=["melspec_2048"]

        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(
            n_fft=2048, hop_length=hop_length, is_db=True
        )
        

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key] # {'melspec_2048_cnt': 14282760192, 'melspec_2048_mean': 6.768444971712967}
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x
    
    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])
        return out

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x, features=self.features) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        x = self.normalize(x)
        x = self.rearrange(x) # -> {'melspec_2048': Tensor{Size([3, 750, 512]) cuda:0 f32}}
        return x['melspec_2048'].permute((0, 2, 1))

if __name__ == "__main__":
    pass