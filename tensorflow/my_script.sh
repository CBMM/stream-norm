#!/bin/bash

python mnist.py --hidden 100 --cell_type LSTM --dau 1 --batch_size 200
#python mnist.py --hidden 114 summaries_dir log_LN/ --cell_type LNGRU --dau 3

python mnist.py --cell_type LNGRU --dau 20 --batch_size 5

python mnist.py --hidden 100 --cell_type LNGRU --dau 20 --batch_size 5
python mnist.py --hidden 100 --cell_type SNGRU --dau 20 --batch_size 5


