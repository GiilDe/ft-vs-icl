#! /bin/sh

#SBATCH --job-name=ft-vs-icl # job name
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2223/giladd/ft-vs-icl/awesome.out # redirect stdout
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2223/giladd/ft-vs-icl/awesome.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-students

#---------${model_name} ${model_arch} ${task} ${k} ${seed} ${perm_id} ${output_path} ${base_dir} ${lr}
bash run.sh en_dense_lm_125m gptmodel_small cb 4 3 0 output_dir base_dir 0.001
