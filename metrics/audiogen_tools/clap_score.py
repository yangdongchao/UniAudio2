# compute_clap_score.py
# Compute CLAP cosine similarity between (audio, text) pairs from:
#   text.scp:  item text...
#   audio_dir: folder containing audio files named like item.wav / item.flac / ...
# Output:
#   out_scp: item \t cos_sim \t abs_cos_sim \t text \t audio_path
# Also prints mean/std/min/max for both cos_sim and abs_cos_sim.

import argparse
from pathlib import Path
import numpy as np
import librosa
import torch
import laion_clap

AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma"]


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype("float32")


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")


def read_text_scp(path: str):
    """text.scp line: item text..."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad line {ln} in {path}: {line}")
            item, text = parts[0], parts[1].strip()
            items.append((item, text))
    return items


def build_audio_index(audio_dir: str, recursive: bool):
    """Map stem -> path. If duplicate stems exist, keep the first found."""
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio_dir not found: {audio_dir}")

    paths = []
    if recursive:
        for ext in AUDIO_EXTS:
            paths.extend(audio_dir.rglob(f"*{ext}"))
    else:
        for ext in AUDIO_EXTS:
            paths.extend(audio_dir.glob(f"*{ext}"))

    idx = {}
    for p in sorted(paths):
        stem = p.stem
        if stem not in idx:
            idx[stem] = p
    return idx


def load_audio_48k_quantized(path: Path, sr: int = 48000) -> torch.Tensor:
    """
    Return Tensor shape: (1, T) float32, quantized via int16 roundtrip (as LAION-CLAP example).
    """
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    wav = wav.reshape(1, -1)
    wav_q = int16_to_float32(float32_to_int16(wav))
    return torch.from_numpy(wav_q).float()


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def get_text_embeds(model, texts, device: torch.device, batch_size: int):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.get_text_embedding(batch, use_tensor=True)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        outs.append(emb.to(device))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def get_audio_embeds_from_data_list(model, audio_tensors, device: torch.device):
    """
    audio_tensors: list[Tensor (1,T)] with variable length.
    We compute one by one to avoid padding issues.
    """
    outs = []
    for x in audio_tensors:
        x = x.to(device)
        emb = model.get_audio_embedding_from_data(x=x, use_tensor=True)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        outs.append(emb.to(device))
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_scp", type=str, required=True)
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--out_scp", type=str, required=True)

    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--sr", type=int, default=48000, help="CLAP expects 48k")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--text_bs", type=int, default=64)
    ap.add_argument("--audio_bs", type=int, default=8)
    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--model_path", type=str, default=None,
                     help="Path to local CLAP model directory or checkpoint file. "
                          "If not specified, will download from internet.")
    args = ap.parse_args()

    # Load pairs
    items = read_text_scp(args.text_scp)
    if args.max_items > 0:
        items = items[:args.max_items]

    audio_index = build_audio_index(args.audio_dir, args.recursive)

    keys, texts, audio_paths = [], [], []
    missing = []
    for k, t in items:
        p = audio_index.get(k)
        if p is None:
            missing.append(k)
            continue
        keys.append(k)
        texts.append(t)
        audio_paths.append(p)

    if missing:
        print(f"[WARN] missing audios for {len(missing)} items (show 10): {missing[:10]}")
    print(f"[INFO] aligned pairs: {len(keys)}")

    if len(keys) == 0:
        raise RuntimeError("No aligned (item, audio, text) pairs found.")

    # Load CLAP
    device = torch.device(args.device)
    model = laion_clap.CLAP_Module(enable_fusion=False)
    
    if args.model_path:
        # Use local model path
        model_path = Path(args.model_path)
        if model_path.is_file():
            # If it's a .pt file, use it directly
            print(f"[INFO] Loading CLAP model from checkpoint file: {model_path}")
            model.load_ckpt(model_path)
        elif model_path.is_dir():
            # If it's a directory, try to find the checkpoint file
            # Common checkpoint names in laion_clap
            checkpoint_names = [
                "630k-audioset-best.pt",
                "630k-best.pt",
                "music_audioset_epoch_15_esc_90.14.pt",
                "music_speech_audioset_epoch_15_esc_89.98.pt",
            ]
            checkpoint_file = None
            for ckpt_name in checkpoint_names:
                ckpt_path = model_path / ckpt_name
                if ckpt_path.exists():
                    checkpoint_file = ckpt_path
                    break
            
            if checkpoint_file:
                print(f"[INFO] Loading CLAP model from: {checkpoint_file}")
                model.load_ckpt(checkpoint_file)
            else:
                # Try to find any .pt file in the directory
                pt_files = list(model_path.glob("*.pt"))
                if pt_files:
                    # Use the first .pt file found (or you can specify which one)
                    checkpoint_file = pt_files[0]
                    print(f"[INFO] Loading CLAP model from: {checkpoint_file}")
                    model.load_ckpt(checkpoint_file)
                else:
                    raise FileNotFoundError(
                        f"No checkpoint file (.pt) found in model directory: {model_path}"
                    )
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    else:
        # Download from internet (original behavior)
        print("[INFO] Loading CLAP model from internet (this may take a while)...")
        model.load_ckpt()
    
    model.to(device)
    model.eval()

    # Text embeddings
    text_emb = get_text_embeds(model, texts, device=device, batch_size=args.text_bs)
    text_emb = l2_normalize(text_emb)

    # Audio embeddings + cosine sim
    scores = torch.empty((len(keys),), device=device, dtype=torch.float32)

    # Optional tqdm
    try:
        from tqdm import tqdm
        batches = tqdm(range(0, len(keys), args.audio_bs), desc="CLAP scoring", unit="batch")
    except Exception:
        batches = range(0, len(keys), args.audio_bs)

    for bi in batches:
        bj = min(bi + args.audio_bs, len(keys))

        # load audio batch
        audio_tensors = []
        for p in audio_paths[bi:bj]:
            audio_tensors.append(load_audio_48k_quantized(p, sr=args.sr))

        audio_emb = get_audio_embeds_from_data_list(model, audio_tensors, device=device)
        audio_emb = l2_normalize(audio_emb)

        sim = (audio_emb * text_emb[bi:bj]).sum(dim=-1)  # cosine similarity
        scores[bi:bj] = sim

    # Write output (include both cos and abs(cos))
    out_path = Path(args.out_scp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, k in enumerate(keys):
            cos = float(scores[i].detach().cpu())
            abs_cos = abs(cos)
            f.write(f"{k}\t{cos:.6f}\t{abs_cos:.6f}\t{texts[i]}\t{audio_paths[i]}\n")

    # Stats
    scores_cpu = scores.detach().float().cpu().numpy()

    print(f"[DONE] wrote: {out_path}")
    print(f"[STATS cos] n={len(scores_cpu)} mean={scores_cpu.mean():.6f} std={scores_cpu.std():.6f} "
          f"min={scores_cpu.min():.6f} max={scores_cpu.max():.6f}")
if __name__ == "__main__":
    main()
