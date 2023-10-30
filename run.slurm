#! /bin/sh

#SBATCH --job-name=ft-vs-icl # job name
#SBATCH --output=/a/home/cc/students/cs/dar/netapp/work/temp/ft-vs-icl/awesome.out # output file
#SBATCH --error=/a/home/cc/students/cs/dar/netapp/work/temp/ft-vs-icl/awesome.err # error file
#SBATCH --partition=killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --gpus=1 # GPUs in total
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process

echo HELLO $SLURM_JOBID

MODEL_SHORT="1_3b"
MODEL_ARCH="gptmodel_large"
MODEL_NAME="en_dense_lm_${MODEL_SHORT}"

#--------------------model_name model_arch task k seed perm_id output_path base_dir lr clip_norm per_layer


while IFS=' ' read -r task num_demonstrations seed perm_id lr clip_norm per_layer; do
  name="${MODEL_SHORT}_${task}_per_layer_${per_layer}_num_demonstrations_${num_demonstrations}_seed_${seed}_perm_id_${perm_id}_lr_${lr}_clip_norm_${clip_norm}"
  echo $name
  echo run.sh
  bash run.sh $MODEL_NAME $MODEL_ARCH $task $num_demonstrations \
  $seed $perm_id output_dir base_dir $lr $clip_norm $per_layer > "artifacts/${name}.out" 2> "artifacts/${name}.err"
  echo run compute_sim.py
  python3 icl_ft/compute_sim.py $task all $MODEL_SHORT ${SLURM_JOBID} > "logs/${name}.txt"
  
  ana_rlt_dir=base_dir/ana_rlt/$MODEL_NAME/${task}_${SLURM_JOBID}
  output_path=output_dir/${SLURM_JOBID}
  save_dir=base_dir/ft_gpt/$task/$MODEL_NAME/${lr}_${SLURM_JOBID}
  rm -r $save_dir
  rm -r $output_path
  rm -r $ana_rlt_dir
done < cmdline_args.txt