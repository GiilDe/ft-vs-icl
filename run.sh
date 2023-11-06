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
            lr=$2
            shift 2
            ;;
        --clip_norm)
            clip_norm=${2}
            shift 2
            ;;
        --per_layer)
            per_layer="$2"
            shift 2
            ;;
        *)
            echo "!!!!! Unknown option: $1 !!!!!"
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
ana_attn=1

# ============= Train FT Model =============
echo "Train FT Model"
k=$icl_k
ana_setting=ft
model_path=$base_dir/gpt_icl/$model_name/model.pt
rm $ana_rlt_dir/ft/record_info.jsonl

optim_group=attn_kv
max_epoch=1
save_dir=$base_dir/gpt_ft/$task/$model_name/${lr}_${SLURM_JOBID}

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

# =========== Evaluate FT, ZS, ICL Models ============
settings="ftzs zs icl"
for ana_setting in $settings; do
    case $ana_setting in
        "ftzs")
            model_path=$save_dir/checkpoint_last.pt
            k=0
            ;;
        "zs")
            model_path=$base_dir/gpt_icl/$model_name/model.pt
            k=0
            ;;
        "icl")
            model_path=$base_dir/gpt_icl/$model_name/model.pt
            k=$icl_k
            ;;
        *)
            echo "Unknown setting: $ana_setting"
            exit 1
            ;;
    esac
    echo "Evaluate $ana_setting setting"
    rm $ana_rlt_dir/$ana_setting/record_info.jsonl
    mkdir -p $ana_rlt_dir/$ana_setting

    bash scripts/ana_validate.sh $seed $task $model_path $arch $k $bsz $ngpu \
        $bpe_path $encoder_path \
        $dict_path $output_path \
        $ana_attn \
        $ana_rlt_dir \
        $ana_setting \
        $perm_id
        
    mv tmp_ana_rlt/${SLURM_JOBID}_${ana_setting}_record_info.jsonl \
        $ana_rlt_dir/$ana_setting/record_info.jsonl
done

rm -r $save_dir
rm -r $output_path