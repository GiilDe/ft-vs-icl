while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --arch)
            arch="$2"
            shift 2
            ;;
        --base_dir)
            base_dir="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --icl_k)
            icl_k="$2"
            shift 2
            ;;
        --output_base_dir)
            output_base_dir="$2"
            shift 2
            ;;
        --perm_id)
            perm_id="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --clip_norm)
            clip_norm="$2"
            shift 2
            ;;
        --per_layer)
            per_layer="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "model_name: $model_name"
echo "arch: $arch"
echo "base_dir: $base_dir"
echo "seed: $seed"
echo "task: $task"
echo "icl_k: $icl_k"
echo "output_base_dir: $output_base_dir"
echo "perm_id: $perm_id"
echo "lr: $lr"
echo "clip_norm: $clip_norm"
echo "per_layer: $per_layer"
echo "============================================"

bsz=1
ngpu=1
bpe_path=$base_dir/gpt_icl/vocab.bpe
encoder_path=$base_dir/gpt_icl/encoder.json
dict_path=$base_dir/gpt_icl/$model_name/dict.txt
output_path=$output_base_dir/${SLURM_JOBID}

ana_rlt_dir=$base_dir/ana_rlt/$model_name/${task}_${SLURM_JOBID}

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
    $perm_id

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
    $perm_id

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
    $perm_id
    
mv tmp_ana_rlt/${SLURM_JOBID}_icl_record_info.jsonl $ana_rlt_dir/icl/record_info.jsonl
