import torch
import re
from collections import defaultdict

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
# ------------------------------------------------------------------
# 1.  load_state() – robustly load a PEFT / LoRA checkpoint
# ------------------------------------------------------------------
def load_state(model: torch.nn.Module,
               ckpt_path: str,
               device: str | torch.device = "cpu",
               strict_shapes: bool = True) -> None:
    """
    Load a checkpoint **only for the parameters that exist in `model`.**
    Handles:
        • DDP prefixes ("module.")
        • PEFT / LoRA prefixes
        • Shape‑mismatch detection
    Args
    ----
    model       your AudioThinking instance
    ckpt_path   path/to/your.ckpt  (either .pt or .bin is fine)
    device      cpu / cuda:n
    strict_shapes  if True, mismatching‑shape tensors are skipped and listed
    """

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:          # → Lightning / Trainer checkpoints
        ckpt = ckpt["model"]

    # -----  Strip common unwanted prefixes  -----
    cleaned_ckpt = {}
    for k, v in ckpt.items():
        new_k = re.sub(r"^module\.", "", k)            # DDP
        cleaned_ckpt[new_k] = v

    # ------------------------------------------------------------------
    # 2.  match keys & shapes
    # ------------------------------------------------------------------
    model_state = model.state_dict()                   # current params
    loadable_state = {}
    mismatched, missing, unexpected = [], [], []

    for k, v in cleaned_ckpt.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                loadable_state[k] = v
            else:
                mismatched.append((k, tuple(model_state[k].shape), tuple(v.shape)))
                if not strict_shapes:
                    loadable_state[k] = v.to(model_state[k].shape)
        else:
            unexpected.append(k)

    for k in model_state.keys():
        if k not in loadable_state:
            missing.append(k)

    # ------------------------------------------------------------------
    # 3.  actually load
    # ------------------------------------------------------------------
    print('loadable_state ', loadable_state.keys())
    msg = (
        f"\n[LoRA‑Loader]  matched={len(loadable_state)}  "
        f"mismatched={len(mismatched)}  "
        f"missing={len(missing)}  "
        f"unexpected={len(unexpected)}"
    )
    print(msg)

    if mismatched:
        print(" →  shape mismatches:")
        for k, mshape, ckshape in mismatched:
            print(f"    {k:80}  model={mshape}  ckpt={ckshape}")

    missing_str = "\n".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
    unexpected_str = "\n".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else "")
    if missing:
        print(f" →  {len(missing)}  keys present in model but missing in ckpt, e.g.:\n{missing_str}")
    if unexpected:
        print(f" →  {len(unexpected)}  keys in ckpt but not used, e.g.:\n{unexpected_str}")

    model.load_state_dict(loadable_state, strict=False)
    print("[LoRA‑Loader]  done.\n")


# ------------------------------------------------------------------
# 4.  verify_state() – sanity check that the tensors really match
# ------------------------------------------------------------------
@torch.no_grad()
def verify_state(model: torch.nn.Module,
                 ckpt_path: str,
                 device: str | torch.device = "cpu") -> None:
    """After load_state(), verify that every **loaded** tensor matches exactly."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    mismatched = defaultdict(float)
    for k, v in model.state_dict().items():
        ck_key = re.sub(r"^module\.", "", k)
        if ck_key in ckpt and ckpt[ck_key].shape == v.shape:
            diff = (v - ckpt[ck_key].to(v.device)).abs().max().item()
            mismatched[diff] += 1

    max_diff = max(mismatched) if mismatched else None
    print(f"[LoRA‑Verify] max  |Δ|  = {max_diff}  (should be 0.0 for exact match)")


# ------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------
if __name__ == "__main__":

    pass
