. ./path.sh # set the path

resume="ReasoningCodec/ep1.checkpoint"
exp_dir="ReasoningCodec"
output_dir="./"
input_dir="wav_path"
music_ssl_mudule_path=modules/our_MERT_BESTRQ/mert_fairseq
python3 infer.py --resume $resume \
                 --exp_dir $exp_dir \
                 --rank 0 \
                 --output_dir $output_dir \
                 --input_dir $input_dir \
                 --music_ssl_mudule_path $music_ssl_mudule_path 



