. ./path.sh
text_tokenizer_path=tools/tokenizer/Text2ID/llama3_2_tokenizer
codec_ckpt='UniAudio2_ckpt/ReasoningCodec.checkpoint'
audio_tokenizer_config='tools/tokenizer/ReasoningCodec_film/infer_config.yaml'
audio_model_path='UniAudio2_ckpt/ReasoningCodec.checkpoint'
exp_dir=./ # set your exp dir
resume=UniAudio2_ckpt/llm_ep3.checkpoint # recommond llm_ep2 or llm_ep3
llm_train_config=UniAudio2_ckpt/llm_config.yaml
codec_config='tools/tokenizer/ReasoningCodec_film/infer_config.yaml'


# TTS
python multi_task_inference.py \
  --task TTS \
  --stage all \
  --text "hello, this is a test case. You can ignore this." \
  --output_dir ./TTS_output \
  --llm_train_config $llm_train_config \
  --exp_dir $exp_dir \
  --resume $resume \
  --text_tokenizer_path $text_tokenizer_path \
  --prompt_text "Convert the given text into natural, human-like speech with clear pronunciation." \
  --audio_tokenizer_config $audio_tokenizer_config \
  --audio_model_path $resume \
  --codec_config $codec_config \
  --codec_ckpt $codec_ckpt \
  --codec_steps 10 


# ASR
python multi_task_inference.py \
  --task ASR \
  --stage all \
  --audio samples/p225_002.wav \
  --output_dir ./ASR_output \
  --llm_train_config $llm_train_config \
  --exp_dir $exp_dir \
  --resume $resume \
  --text_tokenizer_path $text_tokenizer_path \
  --prompt_text "Transcribe the provided audio recording into accurate text." \
  --audio_tokenizer_config $audio_tokenizer_config \
  --audio_model_path $resume \
  --codec_config $codec_config \
  --codec_ckpt $codec_ckpt \
  --codec_steps 10 

# audio caption
python multi_task_inference.py \
  --task audio_caption \
  --stage all \
  --audio samples/sound.wav \
  --output_dir ./caption_output \
  --llm_train_config $llm_train_config \
  --exp_dir $exp_dir \
  --resume $resume \
  --text_tokenizer_path $text_tokenizer_path \
  --prompt_text "audio caption task" \
  --audio_tokenizer_config $audio_tokenizer_config \
  --audio_model_path $resume \
  --codec_config $codec_config \
  --codec_ckpt $codec_ckpt \
  --codec_steps 10 


# text-to-music
python multi_task_inference.py \
  --task TTM \
  --stage all \
  --text "This is a classical music waltz piece played on a glass harp instrument. " \
  --output_dir ./TTM_output \
  --llm_train_config $llm_train_config \
  --exp_dir $exp_dir \
  --resume $resume \
  --text_tokenizer_path $text_tokenizer_path \
  --prompt_text "text-to-music generation" \
  --audio_tokenizer_config $audio_tokenizer_config \
  --audio_model_path $resume \
  --codec_config $codec_config \
  --codec_ckpt $codec_ckpt \
  --codec_steps 10 


#s2t
python multi_task_inference.py \
  --task speech_s2t \
  --stage all \
  --audio samples/p225_002.wav \
  --output_dir ./s2t_output \
  --llm_train_config $llm_train_config \
  --exp_dir $exp_dir \
  --resume $resume \
  --text_tokenizer_path $text_tokenizer_path \
  --prompt_text "speech question with text answer" \
  --audio_tokenizer_config $audio_tokenizer_config \
  --audio_model_path $resume \
  --codec_config $codec_config \
  --codec_ckpt $codec_ckpt \
  --codec_steps 10 

