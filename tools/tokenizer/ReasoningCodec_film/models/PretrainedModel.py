''' The pre-trained model 
    It provides several pre-trained models definition, can be used to extract audio features.
    It should includes two types of output (1) continous audio features (2) discrete tokens
'''
from tools.tokenizer.ReasoningCodec_film.modules.our_MERT_BESTRQ.test import load_best_rq_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tools.tokenizer.ReasoningCodec_film.models.abs_model import AbsPretrainedModel

class BESTRQ_Model(AbsPretrainedModel):
    def __init__(
        self,
        model_dir = 'modules/our_MERT_BESTRQ/mert_fairseq', 
        checkpoint_dir = '', 
        output_features = 'continous',
        layers = [3, 11],
    ):
        super().__init__()
        ''' layers are used to represent the audio. If only includes one layer, 
            we directly use it. If inlcudes multiple layers, we use the mean cross each layers
        '''
        # (1) define the model
        self.model = load_best_rq_model(model_dir = model_dir, checkpoint_dir = checkpoint_dir)
        # (2) fix the model parameter
        self.layers = layers

    def extract_continous_embeds(self, input_audio):
        ''' we assume the input audio is aways 48k binary channels audio
            note that if our pre-trained is not updated during training. We suggest to use contiguous() 
            to clone a new token for following training
        '''
        inputs = self.model(input_audio, features_only = True)
        layer_results = inputs['layer_results']
        if len(self.layers) == 1:
            bestrq_emb = layer_results[self.layers[0]] # B, T, 1024
        else:
            bestrq_emb_0 = []
            for layer in self.layers:
                bestrq_emb.append(layer_results[layer])
            bestrq_emb = torch.stack(bestrq_emb, dim=0).mean(dim=0)  # shape: (L, B, T, D)，L 是层数---> (B, T, D)   
        bestrq_embs = bestrq_emb.permute(0, 2, 1).contiguous() # return B, D, T
        # print("bestrq_embs.shape:",bestrq_embs.shape)
        return bestrq_embs
    
    def extract_continous_embeds_multiple(self, input_audio):
        ''' we assume the input audio is aways 48k binary channels audio
            note that if our pre-trained is not updated during training. We suggest to use contiguous() 
            to clone a new token for following training
        '''
        inputs = self.model(input_audio, features_only = True)
        layer_results = inputs['layer_results']
        if len(self.layers) == 1:
            bestrq_emb = layer_results[self.layers[0]] # B, T, 1024
        else:
            bestrq_emb_acoustic = layer_results[self.layers[0]]
            bestrq_emb_semantic = layer_results[self.layers[1]]
        
        bestrq_emb_acoustic = bestrq_emb_acoustic.permute(0, 2, 1).contiguous() # return B, D, T
        bestrq_emb_semantic = bestrq_emb_semantic.permute(0, 2, 1).contiguous() 
        return bestrq_emb_acoustic, bestrq_emb_semantic
