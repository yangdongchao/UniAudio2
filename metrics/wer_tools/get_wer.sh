input_folder=
input_file=./tmp_others.txt
python3 get_merge.py $input_folder $input_file
python3 get_wer.py $input_file --language en -p 

