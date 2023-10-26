model_name=$1
arch=$2
base_dir=$8

seed=$5
task=$3
bsz=1
ngpu=1
bpe_path=$base_dir/gpt_icl/vocab.bpe
encoder_path=$base_dir/gpt_icl/encoder.json
dict_path=$base_dir/gpt_icl/$model_name/dict.txt
output_path=$7/${SLURM_JOBID}

ana_rlt_dir=$base_dir/ana_rlt/$model_name/${task}_${SLURM_JOBID}

icl_k=$4
perm_id=$6
try_lr=$9
use_linearization=${10}
clip_norm=${11}
per_layer=${12}

# TODO: solve the problem of the following two lines
# rm -r tmp_ana_rlt
# mkdir -p tmp_ana_rlt

# ==================== train the model for analyzing FT setting ============

k=$icl_k
ana_attn=1
ana_setting=ft
model_path=$base_dir/gpt_icl/$model_name/model.pt
rm $ana_rlt_dir/ft/record_info.jsonl

optim_group=attn_kv
lr=$try_lr
max_epoch=1
save_dir=$base_dir/ft_gpt/$task/$model_name/${lr}_${SLURM_JOBID}
# TODO: here too
# rm -r $save_dir
# mkdir -p $save_dir

mkdir -p $ana_rlt_dir/$ana_setting

bash scripts/ana_train.sh $seed $task $model_path $arch $k $bsz $ngpu $bpe_path $encoder_path $dict_path $output_path \
    $ana_attn \
    $ana_rlt_dir \
    $ana_setting \
    $lr \
    $max_epoch \
    $save_dir \
    $optim_group \
    $perm_id \
    $use_linearization \
    $clip_norm \
    $per_layer

mv tmp_ana_rlt/${SLURM_JOBID}_ft_record_info.jsonl $ana_rlt_dir/ft/record_info.jsonl

# ==================== analyzing FT setting ============

k=0
ana_attn=1
ana_setting=ftzs
model_path=$base_dir/ft_gpt/$task/$model_name/${lr}_${SLURM_JOBID}/checkpoint_last.pt
rm $ana_rlt_dir/ftzs/record_info.jsonl

mkdir -p $ana_rlt_dir/$ana_setting

bash scripts/ana_validate.sh $seed $task $model_path $arch $k $bsz $ngpu $bpe_path $encoder_path $dict_path $output_path \
    $ana_attn \
    $ana_rlt_dir \
    $ana_setting \
    $perm_id \
    $use_linearization

mv tmp_ana_rlt/${SLURM_JOBID}_ftzs_record_info.jsonl $ana_rlt_dir/ftzs/record_info.jsonl

# ==================== analyzing ZS setting ============

k=0
ana_attn=1
ana_setting=zs
model_path=$base_dir/gpt_icl/$model_name/model.pt
rm $ana_rlt_dir/zs/record_info.jsonl

mkdir -p $ana_rlt_dir/$ana_setting

bash scripts/ana_validate.sh $seed $task $model_path $arch $k $bsz $ngpu $bpe_path $encoder_path $dict_path $output_path \
    $ana_attn \
    $ana_rlt_dir \
    $ana_setting \
    $perm_id \
    $use_linearization

mv tmp_ana_rlt/${SLURM_JOBID}_zs_record_info.jsonl $ana_rlt_dir/zs/record_info.jsonl

# ==================== analyzing ICL setting ============

k=$icl_k
ana_attn=1
ana_setting=icl
model_path=$base_dir/gpt_icl/$model_name/model.pt
rm $ana_rlt_dir/icl/record_info.jsonl

mkdir -p $ana_rlt_dir/$ana_setting

bash scripts/ana_validate.sh $seed $task $model_path $arch $k $bsz $ngpu $bpe_path $encoder_path $dict_path $output_path \
    $ana_attn \
    $ana_rlt_dir \
    $ana_setting \
    $perm_id \
    $use_linearization
    
mv tmp_ana_rlt/${SLURM_JOBID}_icl_record_info.jsonl $ana_rlt_dir/icl/record_info.jsonl
