#!/usr/bin/env python3
"""
Note that: due to the model suports many tasks. We only give several default tasks here.
Multi-task inference: understanding (audio -> text) and generation (text -> audio tokens -> wav).

Understanding tasks (ASR, Yue_ASR, lyric_recognition, audio_caption, music_caption, audio_understanding, speech_s2t):
  Audio -> ReasoningCodec encode to reason/semantic tokens -> unload Codec -> load LLM -> predict text and save.

Generation tasks (TTS, Yue_TTS, TTA, TTM, LTS, InstructTTS):
  Text -> load LLM -> predict reason/semantic tokens and save -> unload LLM -> load ReasoningCodec -> decode to wav.

Usage examples:
  # Understanding: single audio ASR (tokenize audio first, then LLM prediction)
  python multi_task_inference.py --task ASR --audio /path/to.wav --prompt_json prompts/audio_tasks_prompts.json --output_dir ./out ...

  # Understanding: multiple audios from directory
  python multi_task_inference.py --task ASR --audio_dir /path/to/wavs --prompt_json prompts/audio_tasks_prompts.json ...

  # Understanding: pre-tokenized .pt files, run LLM only
  python multi_task_inference.py --task ASR --reason_pt ./tokens/utt_reason.pt --semantic_pt ./tokens/utt_semantic.pt ...

  # Generation: TTS text -> tokens -> wav (two stages, unload LLM then load Codec in between)
  python multi_task_inference.py --task TTS --text "Hello world." --prompt_json prompts/audio_tasks_prompts.json --stage all ...

  Specify --task (e.g. ASR, TTS, audio_understanding); the script infers understanding vs generation automatically.
"""

import argparse
import glob
import json
import os
import random
import sys
import torch
import torchaudio
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from llm_models.model_new import Model_stage3, ModelArgs
from llm_utils.train_utils import resume_for_inference
from llm_utils.arguments import str2bool


# --------------- Supported tasks ---------------
UNDERSTANDING_TASKS = ["ASR", "Yue_ASR", "lyric_recognition", "audio_caption", "music_caption", "audio_understanding", "speech_s2t"]
GENERATION_TASKS = ["TTS", "Yue_TTS", "TTA", "TTM", "LTS", "InstructTTS", "speech_s2s"]
UNDERSTANDING_TASKS_LOWER = [t.lower() for t in UNDERSTANDING_TASKS]
GENERATION_TASKS_LOWER = [t.lower() for t in GENERATION_TASKS]
TASK_PROMPT_SUFFIX = "\n\n"  # match tokenize_tasks: tokenize(p_norm+'\n\n')

def _prompt_key_from_task(task):
    """Map --task to the key used in prompt_json (aligned with audio_tasks_prompts.json)."""
    t = task.strip().lower()
    if t == "yue_tts":
        return "Yue_TTS"
    if t == "yue_asr":
        return "Yue_ASR"
    if t == "instruct_tts":
        return "InstructTTS"
    if t in ("asr", "tts", "tta", "ttm", "lts"):
        return t.upper()
    if t == "speech_s2s":
        return "speech_s2s"
    if t == "speech_s2t":
        return "speech_s2t"
    return t


def _get_prompt_tensor(args, text_tokenizer, task_name):
    """Get prompt tensor from --prompt_text or --prompt_json, tokenized with TASK_PROMPT_SUFFIX."""
    prompt_text = (getattr(args, "prompt_text", None) or "").strip()
    prompt_json_path = getattr(args, "prompt_json", None)
    if prompt_text:
        chosen = prompt_text
    elif prompt_json_path and os.path.isfile(prompt_json_path):
        with open(prompt_json_path, "r", encoding="utf-8") as f:
            prompts_by_task = json.load(f)
        key = _prompt_key_from_task(task_name)
        if key not in prompts_by_task:
            key = task_name if task_name in prompts_by_task else task_name.upper()
        if key not in prompts_by_task:
            key = list(prompts_by_task.keys())[0]
        prompt_list = prompts_by_task[key]
        if not prompt_list:
            raise ValueError(f"Task '{key}' has no prompts in {prompt_json_path}.")
        chosen = random.choice(prompt_list)
        print(f"[Prompt] task={task_name}, key={key}, chosen: {chosen[:60]}...")
    else:
        raise ValueError("Provide --prompt_text or --prompt_json.")
    text_to_tokenize = chosen.strip() + TASK_PROMPT_SUFFIX
    return torch.tensor(text_tokenizer.tokenize(text_to_tokenize), dtype=torch.long)


def _load_codec(args, device):
    """Load ReasoningCodec_film for encode/decode."""
    from tools.tokenizer.ReasoningCodec_film.reason_tokenizer import ReasoningTokenizer
    music_ssl = getattr(args, "music_ssl_folder", None) or os.path.join(
        _SCRIPT_DIR, "tools", "tokenizer", "ReasoningCodec_film",
        "modules", "our_MERT_BESTRQ", "mert_fairseq"
    )
    if not os.path.isdir(music_ssl):
        music_ssl = os.path.normpath(os.path.join(_SCRIPT_DIR, "tools", "tokenizer", "ReasoningCodec_film", "modules", "our_MERT_BESTRQ", "mert_fairseq"))
    return ReasoningTokenizer(
        train_config=args.codec_config,
        model_path=args.codec_ckpt,
        music_ssl_folder=music_ssl,
        device=device,
    )


def _unload_and_clear_cuda():
    import gc
    torch.cuda.empty_cache()
    gc.collect()


# --------------- Understanding: audio -> tokenize -> LLM -> text ---------------
def _encode_audio_to_tokens(args, device):
    """Encode audio with ReasoningCodec, save *_reason.pt / *_semantic.pt, return (output_dir, list of base names)."""
    codec = _load_codec(args, device)
    sample_rate = codec.sample_rate
    if getattr(args, "audio", None) and os.path.isfile(args.audio):
        audio_list = [(os.path.splitext(os.path.basename(args.audio))[0], args.audio)]
    elif getattr(args, "audio_dir", None) and os.path.isdir(args.audio_dir):
        wavs = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
        audio_list = [(os.path.splitext(os.path.basename(p))[0], p) for p in wavs]
    else:
        raise ValueError("Provide --audio (single wav) or --audio_dir (dir of wavs) for understanding task.")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for name, path in audio_list:
        with torch.no_grad():
            reason_codec, rec_codec = codec.tokenize(path, return_reasoning_text=False)
        # reason_codec, rec_codec: (8, T)
        torch.save(reason_codec.cpu(), os.path.join(out_dir, f"{name}_reason.pt"))
        torch.save(rec_codec.cpu(), os.path.join(out_dir, f"{name}_semantic.pt"))
        names.append(name)
        wave = codec.detokenize_no_reason(rec_codec, return_reasoning_text=False, steps=10)
        #torchaudio.save(os.path.join(out_dir, f"{name}_wave.wav"), wave, sample_rate=sample_rate)
        print(f"[Encode] {path} -> {name}_reason.pt, {name}_semantic.pt")
        #assert 1==2
    return out_dir, names


def _load_config_and_llm(args):
    """Load yaml config and build LLM, restore weights. Return (train_args, model, device)."""
    with open(args.llm_train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    rank = getattr(args, "rank", 0)
    if rank >= 0 and torch.cuda.is_available():
        rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    config = ModelArgs(
        decoder_name=train_args.local_model,
        llm_pretrained_model=train_args.llm_pretrained_model,
        llm_name=train_args.llm_name,
        audio_semantic_vocab_size=train_args.audio_semantic_card,
        audio_reason_vocab_size=train_args.audio_reason_card,
        audio_num_codebooks=train_args.parallel_number - 1,
        audio_embeddings_path=train_args.audio_embeddings_path,
        audio_understanding_expert_path=train_args.audio_understanding_expert_path,
    )
    model = Model_stage3(config)
    model.to(device=device)
    resume_for_inference(args.resume, args.exp_dir, model, device)
    return train_args, model, device


def _get_generator_class(task):
    task = task.strip().lower()
    if task in ("asr", "yue_asr"):
        try:
            from evaluation.asr_task import Generator
        except ImportError:
            from asr_task import Generator
        return Generator
    if task == "lyric_recognition":
        try:
            from evaluation.lyric_asr_task import Generator
        except ImportError:
            from lyric_asr_task import Generator
        return Generator
    if task in ("audio_caption", "music_caption"):
        try:
            from evaluation.audio_music_caption_task import Generator
        except ImportError:
            from audio_music_caption_task import Generator
        return Generator
    if task == "audio_understanding":
        try:
            from evaluation.audio_understanding import Generator
        except ImportError:
            from audio_understanding import Generator
        return Generator
    if task == "speech_s2t":
        try:
            from evaluation.speech_s2t import Generator
        except ImportError:
            from speech_s2t import Generator
        return Generator
    if task in ("tts", "yue_tts"):
        try:
            from evaluation.tts_task import Generator
        except ImportError:
            from tts_task import Generator
        return Generator
    if task == "tta":
        try:
            from evaluation.audiogen_task import Generator
        except ImportError:
            from audiogen_task import Generator
        return Generator
    if task == "ttm":
        try:
            from evaluation.musicgen_task import Generator
        except ImportError:
            from musicgen_task import Generator
        return Generator
    if task == "lts":
        try:
            from evaluation.songen_task import Generator
        except ImportError:
            from songen_task import Generator
        return Generator
    if task == "instruct_tts":
        try:
            from evaluation.insturct_tts_task import Generator
        except ImportError:
            from insturct_tts_task import Generator
        return Generator
    if task == "speech_s2s":
        try:
            from evaluation.speech_s2s import Generator
        except ImportError:
            from speech_s2s import Generator
        return Generator
    raise ValueError(f"Unknown task: {task}. Understanding: {UNDERSTANDING_TASKS}. Generation: {GENERATION_TASKS}.")


def run_understanding(args):
    """Understanding task: optionally tokenize audio first, then load LLM and predict text."""
    task = args.task.strip().lower()
    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}" if args.rank >= 0 and torch.cuda.is_available() else "cpu")

    token_dir = args.output_dir
    names = []

    # If raw audio is provided, encode with Codec and save .pt, then unload Codec
    if (getattr(args, "audio", None) and os.path.isfile(args.audio)) or (getattr(args, "audio_dir", None) and os.path.isdir(args.audio_dir)):
        token_dir, names = _encode_audio_to_tokens(args, device)
        _unload_and_clear_cuda()
    elif getattr(args, "reason_pt", None) and getattr(args, "semantic_pt", None) and os.path.isfile(args.reason_pt) and os.path.isfile(args.semantic_pt):
        token_dir = os.path.dirname(args.reason_pt) or "."
        names = [os.path.basename(args.reason_pt).replace("_reason.pt", "")]
    elif getattr(args, "token_dir", None) and os.path.isdir(args.token_dir):
        reason_files = sorted(glob.glob(os.path.join(args.token_dir, "*_reason.pt")))
        names = [os.path.basename(p).replace("_reason.pt", "") for p in reason_files]
        token_dir = args.token_dir
    else:
        raise ValueError("For understanding task provide --audio/--audio_dir, --reason_pt+--semantic_pt, or --token_dir with *_reason.pt.")

    train_args, model, device = _load_config_and_llm(args)
    GeneratorClass = _get_generator_class(task)
    generator = GeneratorClass(
        model,
        train_args,
        audio_tokenizer_config=args.audio_tokenizer_config,
        audio_model_path=args.audio_model_path,
        text_tokenizer_path=args.text_tokenizer_path,
        is_cfg=getattr(args, "use_cfg", False),
    )
    text_tokenizer = generator._text_tokenizer
    task_prompt = _get_prompt_tensor(args, text_tokenizer, args.task)
    decode_type = getattr(args, "decode_type", "greedy")

    results_path = getattr(args, "results", None) or os.path.join(args.output_dir, f"{task}_results.txt")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    f_out = open(results_path, "w")

    for name in names:
        reason_path = os.path.join(token_dir, f"{name}_reason.pt")
        semantic_path = os.path.join(token_dir, f"{name}_semantic.pt")
        if not os.path.isfile(reason_path) or not os.path.isfile(semantic_path):
            print(f"[Skip] {name}: missing reason/semantic .pt")
            continue
        reason_token = torch.load(reason_path, map_location="cpu")   # (8, T)
        semantic_token = torch.load(semantic_path, map_location="cpu")
        # prepare_asr_task expects (T, 8); Generator uses transpose(0,1)
        reason_token = reason_token.transpose(0, 1).long()   # (T, 8)
        semantic_token = semantic_token.transpose(0, 1).long()

        if task in ("asr", "yue_asr", "lyric_recognition"):
            if decode_type == "beamsearch":
                text_out = generator.generate_asr_beam_search(
                    task_prompt, task_name=task,
                    reason_token=reason_token, semantic_token=semantic_token,
                )
            elif decode_type == "ngram":
                text_out = generator.generate_asr_with_ngram_sampling(
                    task_prompt, task_name=task,
                    reason_token=reason_token, semantic_token=semantic_token,
                    temperature=args.temperature, topk=1, cfg_scale=args.cfg_scale,
                )
            else:
                text_out = generator.generate_asr(
                    task_prompt, task_name=task,
                    reason_token=reason_token, semantic_token=semantic_token,
                    temperature=args.temperature, topk=1, cfg_scale=args.cfg_scale,
                )
        else:
            # audio_caption, music_caption: use generate_audio_caption; audio_understanding: use generate_answer
            if task in ("audio_caption", "music_caption"):
                print('task_prompt ', task_prompt)
                task_pre = text_tokenizer.decode(task_prompt)
                print('task_pre ', task_pre)

                # assert 1==2
                print('reason_token ', reason_token.shape)
                print('semantic_token ', semantic_token.shape)
                text_out = generator.generate_audio_caption(
                    task_prompt, task_name=task,
                    reason_token=reason_token, semantic_token=semantic_token,
                    temperature=args.temperature, topk=1, cfg_scale=args.cfg_scale,
                )
            elif task == "audio_understanding":
                question = (getattr(args, "question", None) or "").strip()
                if not question and getattr(args, "question_file", None) and os.path.isfile(args.question_file):
                    with open(args.question_file, "r", encoding="utf-8") as f:
                        question = f.read().strip()
                if not question:
                    question = "What is described in this audio?"
                q_tokens = text_tokenizer.tokenize(question)
                d = {
                    "text_seq_question": torch.tensor(q_tokens, dtype=torch.long),
                    "reason_seq": reason_token,
                    "semantic_seq": semantic_token,
                }
                keys = ["text_seq_question", "reason_seq", "semantic_seq"]
                types = ["text", "audio", "audio"]
                text_out = generator.generate_answer(
                    task_prompt, task_name=task,
                    d=d, keys=keys, types=types,
                    temperature=args.temperature, topk=1, cfg_scale=args.cfg_scale,
                )
            elif task == "speech_s2t":
                from llm_utils.task_definition import task_formats
                task_format = task_formats["speech_s2t"]
                cond_keys = task_format["keys"][:-1]
                cond_types = task_format["type"][:-1]
                d = {"reason_seq": reason_token, "semantic_seq": semantic_token}
                result = generator.generate_answer(
                    task_prompt=task_prompt,
                    task_name="speech_s2t",
                    d=d,
                    keys=cond_keys,
                    types=cond_types,
                    temperature=args.temperature,
                    topk=args.topk,
                    cfg_scale=args.cfg_scale,
                )
                if result == (-1, -1):
                    print(f"[Skip] {name}: sequence too long for speech_s2t")
                    continue
                text_out = result[0] if isinstance(result, tuple) else result
        f_out.write(f"{name}\t{text_out}\n")
        print(f"[{task}] {name} -> {text_out[:80]}...")
    f_out.close()
    print(f"Results written to {results_path}")


# --------------- Generation: text -> LLM -> token -> Codec decode to wav ---------------
def _generation_method_name(task):
    """Map generation task to Generator method name."""
    t = task.strip().lower()
    if t in ("tts", "yue_tts"):
        return "generate_tts"
    if t == "tta":
        return "generate_audio"
    if t == "ttm":
        return "generate_audio"
    if t == "lts":
        return "generate_LTS"
    if t == "instruct_tts":
        return "generate_instruct_tts"
    if t == "speech_s2s":
        return "generate_audio"
    raise ValueError(f"Unknown generation task: {task}")


def run_generation_stage1(args):
    """Generation stage1: load LLM + task Generator from evaluation, predict reason/semantic tokens, save .pt."""
    task = args.task.strip().lower()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if task == "speech_s2s":
        # speech_s2s: input = source audio (or token_dir with source .pt); output = generated .pt
        from llm_utils.task_definition import task_formats
        cond_format = task_formats.get("speech_s2s")
        if cond_format is None:
            raise ValueError("task_formats has speech_s2s; cannot run speech_s2s.")
        cond_keys = cond_format["keys"][:-2]
        cond_types = cond_format["type"][:-2]

        source_dir = None
        names = []
        if (getattr(args, "audio", None) and os.path.isfile(args.audio)) or (getattr(args, "audio_dir", None) and os.path.isdir(args.audio_dir)):
            device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}" if args.rank >= 0 and torch.cuda.is_available() else "cpu")
            source_dir = os.path.join(out_dir, "source")
            os.makedirs(source_dir, exist_ok=True)
            orig_out = args.output_dir
            args.output_dir = source_dir
            source_dir, names = _encode_audio_to_tokens(args, device)
            args.output_dir = orig_out
            _unload_and_clear_cuda()
        elif getattr(args, "token_dir", None) and os.path.isdir(args.token_dir):
            reason_files = sorted(glob.glob(os.path.join(args.token_dir, "*_reason.pt")))
            names = [os.path.basename(p).replace("_reason.pt", "") for p in reason_files]
            source_dir = args.token_dir
        else:
            raise ValueError("speech_s2s requires --audio, --audio_dir, or --token_dir (source reason/semantic .pt).")

        train_args, model, device = _load_config_and_llm(args)
        GeneratorClass = _get_generator_class(task)
        generator = GeneratorClass(
            model,
            train_args,
            audio_tokenizer_config=args.audio_tokenizer_config,
            audio_model_path=args.audio_model_path,
            text_tokenizer_path=args.text_tokenizer_path,
            is_cfg=getattr(args, "use_cfg", False),
        )
        text_tokenizer = generator._text_tokenizer
        task_prompt = _get_prompt_tensor(args, text_tokenizer, args.task)
        gen_fn = getattr(generator, "generate_audio")

        for name in names:
            reason_path = os.path.join(source_dir, f"{name}_reason.pt")
            semantic_path = os.path.join(source_dir, f"{name}_semantic.pt")
            if not os.path.isfile(reason_path) or not os.path.isfile(semantic_path):
                print(f"[Skip] {name}: missing source {reason_path} or {semantic_path}")
                continue
            reason = torch.load(reason_path, map_location="cpu")
            semantic = torch.load(semantic_path, map_location="cpu")
            d = {
                "reason_seq_1": reason,
                "semantic_seq_1": semantic,
                "reason_seq_2": reason,
                "semantic_seq_2": semantic,
            }
            reason_tokens, semantic_tokens = gen_fn(
                task_prompt=task_prompt,
                task_name="speech_s2s",
                d=d,
                keys=cond_keys,
                types=cond_types,
                temperature=args.temperature,
                topk=args.topk,
                cfg_scale=args.cfg_scale,
            )
            torch.save(reason_tokens.cpu(), os.path.join(out_dir, f"{name}_reason.pt"))
            torch.save(semantic_tokens.cpu(), os.path.join(out_dir, f"{name}_semantic.pt"))
            print(f"[Stage1] speech_s2s {name} -> {name}_reason.pt, {name}_semantic.pt")
        return out_dir

    train_args, model, device = _load_config_and_llm(args)
    GeneratorClass = _get_generator_class(task)
    generator = GeneratorClass(
        model,
        train_args,
        audio_tokenizer_config=args.audio_tokenizer_config,
        audio_model_path=args.audio_model_path,
        text_tokenizer_path=args.text_tokenizer_path,
        is_cfg=getattr(args, "use_cfg", False),
    )
    text_tokenizer = generator._text_tokenizer
    task_prompt = _get_prompt_tensor(args, text_tokenizer, args.task)

    if args.text and args.text.strip():
        items = [("utt_0", args.text.strip())]
    elif args.text_file and os.path.isfile(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            items = [(f"utt_{i}", line.strip()) for i, line in enumerate(f) if line.strip()]
    else:
        raise ValueError("Generation requires --text or --text_file.")
    if not items:
        raise ValueError("No text input.")

    method_name = _generation_method_name(task)
    gen_fn = getattr(generator, method_name)

    for name, text in items:
        text_tokens = torch.tensor(text_tokenizer.tokenize(text), dtype=torch.long)
        kwargs = dict(
            task_prompt=task_prompt,
            task_name=task,
            text_token=text_tokens,
            temperature=args.temperature,
            topk=args.topk,
            cfg_scale=args.cfg_scale,
        )
        if method_name == "generate_instruct_tts":
            kwargs["caption"] = text_tokens
        reason_tokens, semantic_tokens = gen_fn(**kwargs)
        torch.save(reason_tokens.cpu(), os.path.join(out_dir, f"{name}_reason.pt"))
        torch.save(semantic_tokens.cpu(), os.path.join(out_dir, f"{name}_semantic.pt"))
        print(f"[Stage1] {name} -> {name}_reason.pt, {name}_semantic.pt")
    return out_dir


def run_generation_stage2(args):
    """Generation stage2: load ReasoningCodec, decode *_semantic.pt to wav."""
    rank = getattr(args, "rank", 0)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if rank >= 0 and torch.cuda.is_available() else "cpu")
    codec = _load_codec(args, device)
    token_dir = getattr(args, "token_dir", None) or args.output_dir
    reason_files = sorted(glob.glob(os.path.join(token_dir, "*_reason.pt")))
    names = [os.path.basename(p).replace("_reason.pt", "") for p in reason_files]
    wav_dir = getattr(args, "wav_dir", None) or os.path.join(token_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    steps = getattr(args, "codec_steps", 25)
    for name in names:
        semantic_path = os.path.join(token_dir, f"{name}_semantic.pt")
        if not os.path.isfile(semantic_path):
            print(f"[Skip] {name}: missing {semantic_path}")
            continue
        rec_codec = torch.load(semantic_path, map_location=device)
        wave = codec.detokenize_no_reason(rec_codec, return_reasoning_text=False, steps=steps)
        wav_path = os.path.join(wav_dir, f"{name}.wav")
        torchaudio.save(wav_path, wave.cpu(), codec.sample_rate)
        print(f"[Stage2] {name} -> {wav_path}")
    return wav_dir


# --------------- Arguments and entry ---------------
def get_parser():
    p = argparse.ArgumentParser(description="Multi-task inference: understanding (audio->text) or generation (text->wav)")
    p.add_argument("--task", type=str, required=True,
                   help="Understanding: ASR, Yue_ASR, lyric_recognition, audio_caption, music_caption, audio_understanding, speech_s2t. Generation: TTS, Yue_TTS, TTA, TTM, LTS, InstructTTS, speech_s2s.")
    p.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"],
                   help="Generation only: 1=LLM->tokens, 2=codec->wav, all=both (unload LLM in between)")
    # Understanding inputs
    p.add_argument("--audio", type=str, default=None, help="Single wav path (understanding)")
    p.add_argument("--audio_dir", type=str, default=None, help="Dir of wavs (understanding)")
    p.add_argument("--reason_pt", type=str, default=None, help="Pre-tokenized reason.pt (understanding, skip encode)")
    p.add_argument("--semantic_pt", type=str, default=None, help="Pre-tokenized semantic.pt (understanding)")
    p.add_argument("--question", type=str, default=None, help="Question text for audio_understanding")
    p.add_argument("--question_file", type=str, default=None, help="Question per line or single question file")
    # Generation inputs
    p.add_argument("--text", type=str, default="", help="Single input text (generation)")
    p.add_argument("--text_file", type=str, default=None, help="One text per line (generation)")
    # Output
    p.add_argument("--output_dir", type=str, default="./multi_task_out")
    p.add_argument("--results", type=str, default=None, help="Text results path (understanding)")
    p.add_argument("--token_dir", type=str, default=None)
    p.add_argument("--wav_dir", type=str, default=None)
    # Prompt
    p.add_argument("--prompt_text", type=str, default=None)
    p.add_argument("--prompt_json", type=str, default=None, help="e.g. prompts/audio_tasks_prompts.json")
    # LLM
    p.add_argument("--llm_train_config", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--exp_dir", type=str, default=None)
    p.add_argument("--text_tokenizer_path", type=str, default=None)
    p.add_argument("--audio_tokenizer_config", type=str, default=None)
    p.add_argument("--audio_model_path", type=str, default=None)
    p.add_argument("--use_cfg", type=str2bool, default=False)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--decode_type", type=str, default="greedy", choices=["greedy", "ngram", "beamsearch"])
    # Codec
    p.add_argument("--codec_config", type=str, default=None)
    p.add_argument("--codec_ckpt", type=str, default=None)
    p.add_argument("--music_ssl_folder", type=str, default=None)
    p.add_argument("--codec_steps", type=int, default=50)
    p.add_argument("--codec_duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--rank", type=int, default=0)
    return p


def main():
    parser = get_parser()
    args = parser.parse_args()
    task = args.task.strip().lower()

    if task in UNDERSTANDING_TASKS_LOWER:
        has_input = (
            (args.audio and os.path.isfile(args.audio))
            or (args.audio_dir and os.path.isdir(args.audio_dir))
            or (args.reason_pt and args.semantic_pt and os.path.isfile(args.reason_pt) and os.path.isfile(args.semantic_pt))
        )
        if not has_input:
            raise ValueError("For understanding task provide --audio, --audio_dir, or --reason_pt + --semantic_pt.")
        if not args.llm_train_config or not args.text_tokenizer_path:
            raise ValueError("Set --llm_train_config and --text_tokenizer_path.")
        if not (args.prompt_text or (args.prompt_json and os.path.isfile(args.prompt_json))):
            raise ValueError("Set --prompt_text or --prompt_json.")
        if (args.audio or args.audio_dir) and (not args.codec_config or not args.codec_ckpt):
            raise ValueError("For raw audio provide --codec_config and --codec_ckpt to encode first.")
        run_understanding(args)
        return

    if task in GENERATION_TASKS_LOWER:
        has_text = (args.text and args.text.strip()) or (args.text_file and os.path.isfile(args.text_file))
        if not has_text:
            raise ValueError("For generation task provide --text or --text_file.")
        if not args.llm_train_config or not args.text_tokenizer_path:
            raise ValueError("Set --llm_train_config and --text_tokenizer_path.")
        if not (args.prompt_text or (args.prompt_json and os.path.isfile(args.prompt_json))):
            raise ValueError("Set --prompt_text or --prompt_json.")
        args.tts_task_name = _prompt_key_from_task(args.task)
        if args.stage in ("1", "all"):
            run_generation_stage1(args)
            if args.stage == "1":
                print("[Done] Stage 1 only. Run with --stage 2 --token_dir ... to decode to wav.")
                return
            _unload_and_clear_cuda()
            if not args.token_dir:
                args.token_dir = args.output_dir
        if args.stage in ("2", "all"):
            if not args.codec_config or not args.codec_ckpt:
                raise ValueError("For stage 2 or all set --codec_config and --codec_ckpt.")
            if args.stage == "2" and not args.token_dir:
                args.token_dir = args.output_dir
            run_generation_stage2(args)
        print("[Done] Generation pipeline finished.")
        return

    raise ValueError(f"Unsupported task: {task}. Understanding: {UNDERSTANDING_TASKS}. Generation: {GENERATION_TASKS}.")


if __name__ == "__main__":
    main()

