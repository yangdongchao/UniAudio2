# pip install openai-whisper
# pip install editdistance
wav_path=
text_scp=
python whisper_asr.py \
    --audio-dir $wav_path \
    --text-scp $text_scp \
    --batch-size 20 \
    --device cuda \
    --language en \
    --output-scp step_in.scp \
    --output-json step_in.json

python3 json2_wer.py ./step_in.json 
