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


#--------------------model_name model_arch task k seed perm_id output_path base_dir lr use_linearization clip_norm per_layer
#bash run.sh en_dense_lm_125m gptmodel_small cb 4 3 0 output_dir base_dir 0.001 0 0.1
# bash run.sh en_dense_lm_125m gptmodel_small subj 32 4 0 output_dir base_dir 0.003 0

#bash run.sh en_dense_lm_125m gptmodel_small agnews 32 3 0 output_dir base_dir 0.2
#python3 icl_ft/compute_sim.py subj all 1_3b > agnews_125m.txt

name="sst2_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large sst2 32 2 0 output_dir base_dir 0.0005 0 12.0 1 > "${name}.out" 2> "${name}.err"
python3 icl_ft/compute_sim.py sst2 all 1_3b > "${name}.txt"

name="sst5_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large sst5 32 5 0 output_dir base_dir 0.00016 0 12.0 1 > "${name}.out" 2> "${name}.err" 
python3 icl_ft/compute_sim.py sst5 all 1_3b > "${name}.txt"

name="mr_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large mr 32 5 0 output_dir base_dir 0.003 0 12.0 1 > "${name}.out" 2> "${name}.err"
python3 icl_ft/compute_sim.py mr all 1_3b > "${name}.txt"

#bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003 0 7.0 1 > "${name}.out" 2> "${name}.err"
#python3 icl_ft/compute_sim.py subj all 1_3b > subj_1_3b_train_per_layer_clip_norm_7.txt

#bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003 0 9.0 1 > "${name}.out" 2> "${name}.err"
#python3 icl_ft/compute_sim.py subj all 1_3b > subj_1_3b_train_per_layer_clip_norm_9.txt

name="subj_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003 0 12.0 1 > "${name}.out" 2> "${name}.err"
python3 icl_ft/compute_sim.py subj all 1_3b > "${name}.txt"

#bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003 0 1.0 1 > "${name}.out" 2> "${name}.err"
#python3 icl_ft/compute_sim.py subj all 1_3b > subj_1_3b_train_per_layer_clip_norm_different_layers_clipping.txt

#bash run.sh en_dense_lm_1_3b gptmodel_large subj 32 4 0 output_dir base_dir 0.003 0 0.0

name="agnews_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large agnews 32 3 0 output_dir base_dir 0.2 0 12.0 1 > "${name}.out" 2> "${name}.err"
python3 icl_ft/compute_sim.py agnews all 1_3b > "${name}.txt"

name="cb_1_3b_train_per_layer_clip_norm_12"
bash run.sh en_dense_lm_1_3b gptmodel_large cb 32 3 0 output_dir base_dir 0.08 0 12.0 1 > "${name}.out" 2> "${name}.err"
python3 icl_ft/compute_sim.py cb all 1_3b > "${name}.txt"

