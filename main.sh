echo $1	# DEVICE
CUDA_VISIBLE_DEVICES=$1 python source/main.py --rnn_size 768
