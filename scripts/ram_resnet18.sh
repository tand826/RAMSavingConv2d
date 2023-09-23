for size in 1024 2048 4096 8192 10000
do
python check_ram_resnet18.py -s ${size} -m default # only forward
python check_ram_resnet18.py -s ${size} -m default --backward # forward + backward
python check_ram_resnet18.py -s ${size} -m ramsaving # only forward
python check_ram_resnet18.py -s ${size} -m ramsaving --backward # forward + backward
done
