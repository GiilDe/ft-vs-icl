#! /bin/sh

#SBATCH --job-name=ft-vs-icl # job name
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2223/giladd/ft-vs-icl/awesome.out # redirect stdout
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2223/giladd/ft-vs-icl/awesome.err # redirect stderr
#SBATCH --partition=gpu-joberant # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-research


#---------${model_name} ${model_arch} ${task} ${k} ${seed} ${perm_id} ${output_path} ${base_dir} ${lr} ${use_linearization}
# bash run.sh en_dense_lm_125m gptmodel_small cb 4 3 0 output_dir base_dir 0.001


# bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003

# bash run.sh en_dense_lm_125m gptmodel_small subj 32 4 0 output_dir base_dir 0.003

#bash run.sh en_dense_lm_125m gptmodel_small agnews 32 3 0 output_dir base_dir 0.2
#python3 icl_ft/compute_sim.py subj all 1_3b > agnews_125m.txt

bash run.sh en_dense_lm_1_3b gptmodel_large agnews 32 3 0 output_dir base_dir 0.2 1
python3 icl_ft/compute_sim.py agnews all 1_3b > agnews_1_3b_linear.txt

bash run.sh en_dense_lm_1_3b gptmodel_large agnews 32 3 0 output_dir base_dir 0.2 0
python3 icl_ft/compute_sim.py agnews all 1_3b > agnews_1_3b_non_linear.txt

bash run.sh en_dense_lm_1_3b gptmodel_large cb 32 3 0 output_dir base_dir 0.08 1
python3 icl_ft/compute_sim.py cb all 1_3b > cb_1_3b_linear.txt

bash run.sh en_dense_lm_1_3b gptmodel_large cb 32 3 0 output_dir base_dir 0.08 0
python3 icl_ft/compute_sim.py cb all 1_3b > cb_1_3b_non_linear.txt