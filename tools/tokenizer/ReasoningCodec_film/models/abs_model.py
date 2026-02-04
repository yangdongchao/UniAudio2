import torch

class AbsPretrainedModel(torch.nn.Module):
    """
    This is the virtual feature extractor class.
    Other pre-trained model extractor should inherit this class.
    typicially:
        BESQRQ
        WAVLM
        Whisper
        W2VecBERT2
        ...
    """

    @property
    def is_discrete(self):
        """ 
        Return True if the results are discrete token-ids: codec tokens
        Return False if the results are continuous embeddings: e.g., BESTRQ embeddings
        """
        raise NotImplementedError

    def extract_continous_embeds(self, input_audio_0, input_audio_1):
        """
            Return (B, D, T), D denotes the feature dimension
        """
        raise NotImplementedError

    def extract_discrete_tokens(self, input_audio_0, input_audio_1):
        """
            Return (B, D, T), D denotes the codebook layers
        """
        raise NotImplementedError
