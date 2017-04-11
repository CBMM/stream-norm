#!/bin/bash

sbatch --gres=gpu:1 --mem=160000 bash2python.sh --hidden 100 --cell_type SNGRU --dau 1 --batch_size 200
#sbatch --gres=gpu --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type LNGRU --dau 1 --batch_size 200
#sbatch --gres=gpu --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type LSTM --dau 1 --batch_size 200
#sbatch --gres=gpu --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type GRU --dau 1 --batch_size 200

# sbatch --gres=gpu:1 --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type SNGRU --dau 1 --batch_size 200
# sbatch --gres=gpu:1 --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type SNGRU --dau 100 --batch_size 2
# sbatch --gres=gpu:1 --mem=160000 -n 4 bash2python.sh --hidden 100 --cell_type SNGRU --dau 200 --batch_size 1
