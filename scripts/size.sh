python check_size.py -m default
python check_size.py -m default --backward
python check_size.py -m ramsaving # only forward
python check_size.py -m ramsaving --backward --skip_input_grad # forward + backward (without input grad)
python check_size.py -m ramsaving --backward # forward + backward