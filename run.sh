while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpu)
            ngpu="$2"
            shift 2
            ;;
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
echo "ngpu: $ngpu"
echo "============================================"

bpe_path=$base_dir/gpt_icl/vocab.bpe
encoder_path=$base_dir/gpt_icl/encoder.json
dict_path=$base_dir/gpt_icl/$model_name/dict.txt
output_path=$output_base_dir/${SLURM_JOBID}
activations_dir=$base_dir/activations/$model_name/${task}_${SLURM_JOBID}
model_path=$base_dir/gpt_icl/$model_name/model.pt
save_dir=$base_dir/gpt_ft/$task/$model_name/${lr}_${SLURM_JOBID}

bsz=1
analyze_attn=1

mkdir -p $output_path
echo  "output dir: $output_path"

# ============= Train FT Model =============
echo "Train FT Model"
analysis_setting=ft
optim_group=attn_kv
max_epoch=1

rm $activations_dir/ft/record_info.jsonl
mkdir -p $activations_dir/$analysis_setting

python3 scripts/trainer.py - \
    --task fs_eval \
    --tokens-per-sample 2048  \
    --criterion fs_ft \
    --arch $arch  \
    --gpt2-vocab-bpe $bpe_path  \
    --gpt2-encoder-json $encoder_path \
    --log-format simple  \
    --required-batch-size-multiple 1 \
    --log-interval 1 \
    --warmup-updates 0 \
    --optimizer sgd \
    --lr $lr \
    --clip-norm $clip_norm \
    --max-epoch $max_epoch \
    --curriculum 1000000 \
    --max-update 1000000 \
    --fp16 \
    --eval-data $task \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed $seed \
    --reset-dataloader \
    --k $icl_k \
    --batch-size $bsz \
    --batch-size-valid $bsz \
    --ddp-backend=c10d \
    --gpt-dict $dict_path \
    --gpt-model-path $model_path \
    --analyze-attn $analyze_attn \
    --activations-dir $activations_dir \
    --analysis-setting $analysis_setting \
    --save-dir $save_dir \
    --save-interval $max_epoch \
    --save-interval-updates 1000000 \
    --validate-interval 1000000 \
    --disable-validation \
    --uid $SLURM_JOBID \
    --optim-group $optim_group \
    --distributed-world-size $ngpu \
    --per-layer $per_layer \
    --permut-index $perm_id |& tee $output_path/train_log_$analysis_setting.txt

# =========== Evaluate FT, ZS, ICL Models ============
n_classes=2 # case sst2, mr, subj
if [ "$task" = "agnews" ]
then
n_classes=4
fi
if [ "$task" = "sst5" ]
then
n_classes=5
fi
if [ "$task" = "cb" ]
then
n_classes=3
fi

bsz_eval=$((n_classes * bsz))
settings="ftzs zs icl"

for analysis_setting in $settings; do
    case $analysis_setting in
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
            echo "Unknown setting: $analysis_setting"
            exit 1
            ;;
    esac
    echo "Evaluate $analysis_setting setting"
    rm $activations_dir/$analysis_setting/record_info.jsonl
    mkdir -p $activations_dir/$analysis_setting

    python3 scripts/trainer.py - \
    --task fs_eval \
    --tokens-per-sample 2048  \
    --criterion fs_eval \
    --arch $arch  \
    --gpt2-vocab-bpe $bpe_path  \
    --gpt2-encoder-json $encoder_path \
    --log-format simple  \
    --max-epoch 1 \
    --required-batch-size-multiple 1 \
    --log-interval 1 \
    --warmup-updates 0 \
    --optimizer sgd \
    --max-update 0 \
    --fp16 \
    --eval-data $task \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed $seed \
    --reset-dataloader \
    --no-save \
    --k $k \
    --batch-size $bsz_eval \
    --batch-size-valid $bsz_eval \
    --ddp-backend=c10d \
    --gpt-dict $dict_path \
    --gpt-model-path $model_path \
    --analyze-attn $analyze_attn \
    --activations-dir $activations_dir \
    --analysis-setting $analysis_setting \
    --uid $SLURM_JOBID \
    --distributed-world-size $ngpu \
    --permut-index $perm_id |& tee $output_path/train_log_$analysis_setting.txt
        
    mv artifacts/tmp_activations/${SLURM_JOBID}_${analysis_setting}_record_info.jsonl \
        $activations_dir/$analysis_setting/record_info.jsonl
done

rm -r $save_dir
rm -r $output_path