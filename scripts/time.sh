for size in 1000 2000 4000 8000 16000 22182
do
python check_time.py -s ${size} -m default # only forward
python check_time.py -s ${size} -m default --backward # forward + backward
python check_time.py -s ${size} -m ramsaving # only forward
python check_time.py -s ${size} -m ramsaving --backward --skip_input_grad # forward + backward (without input grad)
python check_time.py -s ${size} -m ramsaving --backward # forward + backward
done