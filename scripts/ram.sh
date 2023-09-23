for size in 1024 2048 4096 8192 16384 22182
do
python check_ram.py -s ${size} -m default # only forward
python check_ram.py -s ${size} -m default --backward # forward + backward
python check_ram.py -s ${size} -m ramsaving # only forward
python check_ram.py -s ${size} -m ramsaving --backward --skip_input_grad # forward + backward (without input grad)
python check_ram.py -s ${size} -m ramsaving --backward # forward + backward
done