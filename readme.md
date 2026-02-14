<p align="center">
    <a href="https://dongchaoyang.top/UniAudio2Demo/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="https://arxiv.org/pdf/2602.04683">Paper</a>
    <br>
    <a href="https://huggingface.co/Dongchao/UniAudio2_ckpt">Checkpoints 🤗</a> 
    </picture></a>
    <br>
</p>

---

# UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization

UniAudio 2.0 is a unified audio foundation model for speech, sound, and music. It uses ReasoningCodec (reasoning tokens and reconstruction tokens) and a unified autoregressive architecture trained on 100B text and 60B audio tokens.
The overview of UniAudio 2.0 as following picture shows.
![The overview of UniAudio 2.0](fig/overview.png)


## Key Features

- ReasoningCodec: discrete audio codec with reasoning tokens and reconstruction tokens
- Unified autoregressive model over text and audio
- Multi-stage training and multi-task data
- Strong in-domain and few-shot/zero-shot performance

## Supported Tasks (included but not limited to the following tasks)

- Speech: TTS (EN/ZH/Yue), Audio-Instructed TTS, InstructTTS, ASR, Dysarthric Speech Recognition, S2S Q&A, S2T Q&A
- Sound: Text-to-Sound, Audio Caption, audio-question answer
- Music: Song Generation (EN/ZH) and Recognition, Text-to-Music Generation, music-question answer


## Installation

**From source (recommended)**

```bash
# 1. Clone the repo and enter the directory
git clone https://github.com/yangdongchao/UniAudio2

# 2. Create environment (Python 3.10)
conda create -n uniaudio2 python=3.10
conda activate uniaudio2

# 3. Editable install: installs dependencies and links this project so imports work
pip install -e .

```

## Quick Start

All tasks are run via **`multi_task_inference.py`**. You need to prepare:

- download the checkpoints from HuggingFace https://huggingface.co/Dongchao/UniAudio2_ckpt

- Note that, we train two version codec (codebook size=1024, and codebook size=8192 (reasoning branch is 4096)). Our currently LLM support the 8192 version (**`ReasoningCodec.checkpoint`**). The **`ReasoningCodec_1024.checkpoint`** version can be used as reconstruction task comparison with other models.

- For the LLM checkpoints, we recommend to use **`llm_ep2.checkpoint`** or **`llm_ep3.checkpoint`**. 

- Update the **`tools/tokenizer/ReasoningCodec_film/codec_infer_config.yaml`**, use the right path based on your download model path

- Run the following code (refer to **`test.sh`**) to test different tasks

- Note that, we donot use text instruction data to train this model, so the instruction understanding ability may limited. You can change your prompt to adjust it if you are not satisfaction the performance. 

### Understanding (audio → text)

Input: one wav, a directory of wavs, or pre-encoded `*_reason.pt` + `*_semantic.pt`. Output: text in `{output_dir}/{task}_results.txt`.

**ASR**
```bash
python multi_task_inference.py \
  --task ASR \
  --audio samples/p225_002.wav \
  --output_dir ./ASR_output \
  --llm_train_config <LLM_CONFIG> \
  --exp_dir <EXP_DIR> \
  --resume <RESUME> \
  --text_tokenizer_path tools/tokenizer/Text2ID/llama3_2_tokenizer \
  --prompt_text "Transcribe the provided audio recording into accurate text." \
  --audio_tokenizer_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml \
  --codec_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml \
  --codec_ckpt <CODEC_CKPT>
```

**Audio caption**
```bash
python multi_task_inference.py --task audio_caption \
  --audio samples/sound.wav --output_dir ./caption_output \
  --llm_train_config <LLM_CONFIG> --exp_dir <EXP_DIR> --resume <RESUME> \
  --text_tokenizer_path tools/tokenizer/Text2ID/llama3_2_tokenizer \
  --prompt_text "Describe the audio content." \
  --audio_tokenizer_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml \
  --codec_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml --codec_ckpt <CODEC_CKPT>
```

**Speech-S2T (speech question → text answer)**  
Same as above with `--task speech_s2t` and a suitable `--prompt_text` (e.g. "speech question with text answer").

**Multiple files**  
Use `--audio_dir /path/to/wavs` instead of `--audio`.  

### Generation (text → audio)

Input: `--text "..."` or `--text_file list.txt`. Output: `*_reason.pt` / `*_semantic.pt` and, with `--stage all`, decoded wavs in `{output_dir}/wavs/`.

**TTS**
```bash
python multi_task_inference.py \
  --task TTS \
  --stage all \
  --text "Hello, this is a test." \
  --output_dir ./TTS_output \
  --llm_train_config <LLM_CONFIG> --exp_dir <EXP_DIR> --resume <RESUME> \
  --text_tokenizer_path tools/tokenizer/Text2ID/llama3_2_tokenizer \
  --prompt_text "Convert the given text into natural speech." \
  --audio_tokenizer_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml \
  --codec_config tools/tokenizer/ReasoningCodec_film/infer_config.yaml \
  --codec_ckpt <CODEC_CKPT> --codec_steps 10
```

**Text-to-Music (TTM)**  
Same pattern with `--task TTM` and e.g. `--text "A classical waltz on glass harp."` and `--prompt_text "text-to-music generation"`.
Our model also support text-to-song generation, but it can only support 30s song generation. If you want to use better song generation performance, please refer to our <a href="https://heartmula.github.io/"> HeartMula</a> 

**Stages**  
- `--stage all`: run LLM to get tokens, then decode to wav (default).  
- `--stage 1`: only save `*_reason.pt` / `*_semantic.pt`.  
- `--stage 2`: decode existing `*_semantic.pt` in `--token_dir` to wav (set `--token_dir` and `--codec_*`).

### Using a prompt JSON

Instead of `--prompt_text`, you can use a JSON that maps task names to prompt lists (one is sampled at runtime):

```bash
--prompt_json prompts/audio_tasks_prompts.json
```

See `prompts/audio_tasks_prompts.json` for the expected keys (e.g. `ASR`, `TTS`, `audio_caption`, `speech_s2t`).


## Acknowledgements

I am deeply grateful to all co-authors and collaborators of the UniAudio series for their collaboration, trust, and inspiring discussions.  
Their contributions across modeling, datasets, training infrastructure, and evaluation have been fundamental to the development of UniAudio series.


## Citation

If you use UniAudio 2.0, please cite:

    @article{uniaudio2,
      title={UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization},
      author={Dongchao Yang, Yuanyuan Wang, Dading Chong, Songxiang Liu, Xixin Wu, Helen Meng},
      year={2026}
    }

## License

MIT License.
