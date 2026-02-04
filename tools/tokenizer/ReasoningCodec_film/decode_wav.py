import os
import glob
import math
import argparse
import numpy as np
import torch
import torchaudio
import yaml

from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.ReasoningCodec_film.models.AudioDiffusion1D import AudioDiffusion1D
from tools.tokenizer.ReasoningCodec_film.models.model_utils import load_state
from tools.tokenizer.common import VolumeNorm
from tools.tokenizer.ReasoningCodec_film.models.scalar24k import ScalarAE
from transformers import WhisperFeatureExtractor


class ReasoningTokenizer(AbsTokenizer):
    def __init__(self, train_config, model_path, device=torch.device('cpu')):
        super(ReasoningTokenizer, self).__init__()
        with open(train_config, "r", encoding="utf-8") as f:
            train_args = yaml.safe_load(f)
            train_args = argparse.Namespace(**train_args)

        self.sample_rate = 24000
        self.device = device
        self.MAX_DURATION = 360
        self.n_codebook = 8
        self.sq_codec_hz = 25
        self.rec_frame_rate = 12.5
        self.reason_frame_rate = 5

        self.SQCodec = ScalarAE(scalar_config=train_args.sq_config, resume_path=train_args.sq_resume)
        self.SQCodec = self.SQCodec.eval().to(device)

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(train_args.whisper_path)
        self.transfer16k = torchaudio.transforms.Resample(24000, 16000).to(device)

        self.model = AudioDiffusion1D(
            num_channels=train_args.num_channels,
            pre_trained_model_name='whisper&bestrq',
            features_type='continuous',
            vq_training=True,
            unet_model_name=train_args.unet_model_name,
            unet_model_config_path=train_args.transformer_diffusion_config,
            whisper_path=train_args.whisper_path,
            reason_lm_path=train_args.reason_lm_path,
            best_rq_ckpt=train_args.best_rq_ckpt,
            llm_path=train_args.llm_path,
            prompt_path=train_args.prompt_path,
            uncondition=True,
            use_detokenizer=True
        )

        if getattr(train_args, "reconstruction_path", None) is not None:
            load_state(self.model, train_args.reconstruction_path)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu')['model']
            state_dict = {k.split("module.")[-1] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(device)
        print("Successfully loaded checkpoint from:", model_path)
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)

        self.volume_norm = VolumeNorm(params=[-16, 3], sample_rate=24000, energy_threshold=1e-6)

    @torch.no_grad()
    def code2sound_no_reason(self, rec_codec, return_reasoning_text, duration=20, guidance_scale=1.5, num_steps=20, disable_progress=False):
        rec_codec = rec_codec.to(self.device)
        first_latent = torch.randn(rec_codec.shape[0], int(duration * 25), 136).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0

        min_samples = int(duration * self.rec_frame_rate)
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples // 2

        rec_codes_len = rec_codec.shape[-1]
        target_len = int((rec_codes_len - first_latent_codes_length) / 12.5 * self.sample_rate)

        if rec_codes_len < min_samples:
            while rec_codec.shape[-1] < min_samples:
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:, :, 0:min_samples]

        rec_codes_len = rec_codec.shape[-1]
        if (rec_codes_len - ovlp_samples) % hop_samples > 0:
            len_codes = math.ceil((rec_codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while rec_codec.shape[-1] < len_codes:
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:, :, 0:len_codes]

        latent_length = int(duration * self.sq_codec_hz)
        latent_list = []
        spk_embeds = None

        use_cuda_autocast = (isinstance(self.device, torch.device) and self.device.type == "cuda")
        if use_cuda_autocast:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.no_grad()

        with autocast_ctx:
            for sinx in range(0, rec_codec.shape[-1] - hop_samples, hop_samples):
                codes_input = [rec_codec[:, :, sinx:sinx + min_samples]]

                if sinx == 0:
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(
                        codes_input, spk_embeds, first_latent, latent_length, incontext_length,
                        additional_feats=[], guidance_scale=guidance_scale,
                        num_steps=num_steps, disable_progress=disable_progress, scenario='other_seg'
                    )
                    latent_list.append(latents)
                else:
                    true_latent = latent_list[-1][:, -ovlp_frames:, :]
                    len_add = latent_length - true_latent.shape[1]
                    incontext_length = true_latent.shape[1]
                    true_latent = torch.cat(
                        [true_latent, torch.randn(true_latent.shape[0], len_add, true_latent.shape[-1]).to(self.device)],
                        1
                    )
                    latents = self.model.inference_codes(
                        codes_input, spk_embeds, true_latent, latent_length, incontext_length,
                        additional_feats=[], guidance_scale=guidance_scale,
                        num_steps=num_steps, disable_progress=disable_progress, scenario='other_seg'
                    )
                    latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:, first_latent_length:, :]

        min_samples_wav = int(duration * self.sample_rate)
        hop_samples_wav = min_samples_wav // 4 * 3
        ovlp_samples_wav = min_samples_wav - hop_samples_wav

        with torch.no_grad():
            output = None
            for latent in latent_list:
                cur_output = self.SQCodec.decode(latent.transpose(1, 2)).squeeze(0)
                cur_output = cur_output[:, 0:min_samples_wav].detach().cpu()

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples_wav)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples_wav:] = (
                        output[:, -ovlp_samples_wav:] * ov_win[:, -ovlp_samples_wav:]
                        + cur_output[:, 0:ovlp_samples_wav] * ov_win[:, 0:ovlp_samples_wav]
                    )
                    output = torch.cat([output, cur_output[:, ovlp_samples_wav:]], -1)

            output = output[:, 0:target_len]
        return output

    def detokenize_no_reason(self, rec_codec, return_reasoning_text, min_duration=30, steps=50, guidance_scale=1.5, disable_progress=False):
        wave = self.code2sound_no_reason(
            rec_codec.unsqueeze(0),
            return_reasoning_text=return_reasoning_text,
            duration=min_duration,
            guidance_scale=guidance_scale,
            num_steps=steps,
            disable_progress=disable_progress
        )
        return wave


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, required=True, help='the config file')
    parser.add_argument('--audio_tokenizer_path', type=str, required=True, help='tokenizer checkpoint path')

    # ✅ 单输入目录：里面同时有 *_semantic.pt 和 *_gt.pt
    parser.add_argument('--pt_path', type=str, required=True, help='folder containing both *_semantic.pt and *_gt.pt')

    # 输出：base dir；默认会写到 save_dir/semantic 和 save_dir/gt
    parser.add_argument('--save_dir', type=str, required=True, help='base output dir')
    parser.add_argument('--semantic_save_dir', type=str, default=None, help='output dir for semantic decode')
    parser.add_argument('--gt_save_dir', type=str, default=None, help='output dir for gt decode')

    # 只解哪一类
    parser.add_argument('--decode_types', type=str, default='semantic', choices=['semantic', 'gt', 'all'])

    # 防止同名覆盖（同一个 base id，同时有 semantic/gt）
    parser.add_argument('--keep_suffix_in_name', action='store_true',
                        help='output wav as xxx_semantic.wav / xxx_gt.wav')

    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU (rank-1 convention kept)')
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    return parser


def _decode_one_type(tokenizer, device, pt_path, save_dir, suffix, steps, duration, guidance_scale, keep_suffix_in_name):
    pattern = os.path.join(pt_path, f"*{suffix}")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print(f"[WARN] No files matched: {pattern}")
        return 0

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Decoding {len(files)} files: {suffix}  |  {pt_path} -> {save_dir}")

    n_ok = 0
    for f in files:
        try:
            rec_codec = torch.load(f, map_location='cpu').long()
            # if 'gt' in suffix:
            #     rec_codec = rec_codec.transpose(0,1)
            base = os.path.basename(f).replace(suffix, "")
            if keep_suffix_in_name:
                out_wav = f"{base}{suffix.replace('.pt','')}.wav"  # xxx_semantic.wav / xxx_gt.wav
            else:
                out_wav = f"{base}.wav"  # 可能覆盖（不推荐）

            with torch.no_grad():
                wav = tokenizer.detokenize_no_reason(
                    rec_codec.to(device),
                    return_reasoning_text=False,
                    min_duration=duration,
                    steps=steps,
                    guidance_scale=guidance_scale
                )

            torchaudio.save(os.path.join(save_dir, out_wav), wav.detach().cpu(), sample_rate=24000)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}")

    return n_ok


if __name__ == '__main__':
    args = get_parser().parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 保持你原来的 rank-1 习惯
    args.rank = args.rank - 1
    if args.rank >= 0 and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}')
    else:
        device = torch.device('cpu')
        args.rank = -1

    tokenizer = ReasoningTokenizer(
        model_path=args.audio_tokenizer_path,
        train_config=args.train_config,
        device=device
    )

    semantic_out = args.semantic_save_dir or os.path.join(args.save_dir, "semantic")
    gt_out = args.gt_save_dir or os.path.join(args.save_dir, "gt")

    total = 0
    if args.decode_types in ("semantic", "all"):
        total += _decode_one_type(
            tokenizer, device, args.pt_path, semantic_out,
            suffix="_semantic.pt",
            steps=args.steps, duration=args.duration, guidance_scale=args.guidance_scale,
            keep_suffix_in_name=args.keep_suffix_in_name
        )

    if args.decode_types in ("gt", "all"):
        total += _decode_one_type(
            tokenizer, device, args.pt_path, gt_out,
            suffix="_gt.pt",
            steps=args.steps, duration=args.duration, guidance_scale=args.guidance_scale,
            keep_suffix_in_name=args.keep_suffix_in_name
        )

    print(f"[DONE] decoded wavs: {total}")
