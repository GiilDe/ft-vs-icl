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
#SBATCH --cpus-per-task=16 # CPU cores per process

DEBUG_MODE=true
echo HELLO $SLURM_JOBID

MODEL_SHORT="1_3b"
MODEL_ARCH="gptmodel_large"
MODEL_NAME="en_dense_lm_${MODEL_SHORT}"

output_base_dir="artifacts/output_dir"
base_dir="artifacts"
log_dir="artifacts/logs"
debug_dir="artifacts/debug"

bsz=1
ngpu=1

mkdir -p $base_dir
mkdir -p $output_base_dir
mkdir -p $log_dir/$MODEL_NAME
mkdir -p $debug_dir
mkdir -p $base_dir/activations
mkdir -p $base_dir/tmp_activations

while IFS=' ' read -r task num_demonstrations seed perm_id lr clip_norm per_layer model_trained; do
  name="${MODEL_NAME}/${task}_per_layer_${per_layer}_k_${num_demonstrations}_seed_${seed}_perm_${perm_id}_lr_${lr}_bsz_${bsz}_clip_norm_${clip_norm}_model_trained_${model_trained}"
  echo $name

  echo run.sh
  bash run.sh --model_name $MODEL_NAME --arch $MODEL_ARCH \
  --task $task --icl_k $num_demonstrations \
  --seed $seed --perm_id $perm_id --output_base_dir $output_base_dir --base_dir $base_dir \
  --lr $lr --clip_norm $clip_norm --ngpu $ngpu --bsz $bsz --uid $SLURM_JOBID \
  --per_layer $per_layer --model_trained $model_trained > "$debug_dir/${name}.out" 2> "$debug_dir/${name}.err"
 
  echo run metrics.py
  python3 scripts/metrics.py --task $task --mode all \
    --model $MODEL_SHORT --uid ${SLURM_JOBID} \
    --base_dir $base_dir --name $name > "$log_dir/${name}.txt"  
  activations_dir=$base_dir/activations/$MODEL_NAME/${task}_${SLURM_JOBID}
  
  if [ "$DEBUG_MODE" = true ]; then
    echo "DEBUG MODE: does not remove $activations_dir"
  else
    rm -r $activations_dir
  fi
  
done < $1