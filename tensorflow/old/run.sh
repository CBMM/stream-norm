#!/bin/bash

sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_SN1/ --cell_type SNGRU --dau 1
sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_SN2/ --cell_type SNGRU --dau 2
sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_SN3/ --cell_type SNGRU --dau 4

sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_LN1/ --cell_type LNGRU --dau 1

sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_LN2/ --cell_type LNGRU --dau 2
sbatch --mem=30000 -n 2 bash2python.sh --hidden 128 summaries_dir log_LN3/ --cell_type LNGRU --dau 4

