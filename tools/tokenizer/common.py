import torch
import random
import torch.nn as nn
from torchaudio import transforms as T

def clip_by_length(x, length, factor):

    if len(x) <= length:
        return x

    start = random.randint(0, len(x) - length - 1)
    start = start // factor * factor
    x = x[start: start + length]
    return x

def speech_edit_find_time_stamp(x, token_list):
    assert isinstance(x, torch.Tensor)
    x, counts = torch.unique_consecutive(x, return_counts=True)
    x = [token_list[i.item()] for i in x]
    counts = torch.cumsum(counts, dim=0)
    counts = counts.cpu().tolist()

    # Possible Phones obtained from kaldi: 
    # (B)egin, (E)nd, (I)nternal and (S)ingleton 
    # & SIL & SPN_S
    # The phone_table doesn't contain SPN_S so it is replaced by <UNK>
    ans, buf = [], []
    for phone, count in zip(x, counts):
        if phone.endswith('_B') or phone.endswith('_I') or phone.endswith("_E"):
            buf.append((phone, count))
            if phone.endswith("_E"):
                phone_seq = tuple([x[0] for x in buf])
                count = buf[-1][1]
                ans.append((phone_seq, count))
                buf = []
        elif phone == "SIL" or phone.endswith('_S'):
            ans.append((phone, count))
        else:
            ans.append((phone, count)) # usually  SPN_S

    # If too short, mask it all.
    if len(ans) <= 2:
        return (0, ans[-1][1])

    num = random.randint(1, 2) # mask 1-2 words
    word_start = random.randint(0, len(ans) - num)

    if word_start == 0:
        start = 0
    else:
        start = ans[word_start - 1][1]
        
    end = ans[word_start + num - 1][1]

    return (start, end)

def codec_specaug(codec, mask_id):
    """  
    Simply specaug on codec audio input.
    Apply time mask with max-width 5% of the total length; 10 masks
    Apply codec (frequency) mask with only 0 / 1 bin. 1 mask.
    """
    T, D = codec.size()
    max_len = int(T * 0.05)

    for i in range(5):
        start = random.randint(0, T - max_len - 1)
        length = random.randint(0, max_len)
        codec[start: start + length] = mask_id

    if random.random() > 1.0:
        dim = random.randint(0, D - 1)
        codec[:, dim] = mask_id

    return codec.view(-1).contiguous()
    

def fix_and_load_json(s):
    # Remove trailing commas before } or ]
    s = re.sub(r',(\s*[}\]])', r'\1', s)

    # Insert missing commas between properties
    # Match positions where a value is followed by a newline and then a quote without a comma
    pattern = r'(?<=[}\]0-9truefalsenull"])\s*(\n\s*)"'
    replacement = r',\1"'
    s = re.sub(pattern, replacement, s)

    # Now try to parse the JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after fixing: {e}")

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 3], sample_rate=24000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal