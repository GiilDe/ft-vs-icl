#! /bin/sh

#SBATCH --job-name=ft-vs-icl # job name
#SBATCH --output=/a/home/cc/students/cs/dar/netapp/work/temp/ft-vs-icl/awesome.out # output file
#SBATCH --error=/a/home/cc/students/cs/dar/netapp/work/temp/ft-vs-icl/awesome.err # error file
#SBATCH --partition=killable
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --gpus=4 # GPUs in total
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --mem=100000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process

echo HELLO $SLURM_JOBID

MODEL_SHORT="1_3b"
MODEL_ARCH="gptmodel_large"
MODEL_NAME="en_dense_lm_${MODEL_SHORT}"

output_base_dir="artifacts/output_dir"
base_dir="artifacts/base_dir"
log_dir="artifacts/logs"
debug_dir="artifacts/debug"

ngpu=1

while IFS=' ' read -r task num_demonstrations seed perm_id lr clip_norm per_layer; do
  name="${MODEL_SHORT}_${task}_per_layer_${per_layer}_num_demonstrations_${num_demonstrations}_seed_${seed}_perm_id_${perm_id}_lr_${lr}_clip_norm_${clip_norm}"
  echo $name
  
  echo run.sh
  bash run.sh --model_name $MODEL_NAME --arch $MODEL_ARCH \
  --task $task --icl_k $num_demonstrations \
  --seed $seed --perm_id $perm_id --output_base_dir $output_base_dir --base_dir $base_dir \
  --lr $lr --clip_norm $clip_norm --ngpu $ngpu \
  --per_layer $per_layer > "$debug_dir/${name}.out" 2> "$debug_dir/${name}.err"
 
  echo run analysis.py
  python3 scripts/analysis.py $task all $MODEL_SHORT ${SLURM_JOBID} $base_dir > "$log_dir/${name}.txt"  
  analysis_results_dir=$base_dir/analysis_results/$MODEL_NAME/${task}_${SLURM_JOBID}
  
  rm -r $analysis_results_dir
done < $1