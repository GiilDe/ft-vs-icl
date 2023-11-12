import os
import json
import numpy as np
import jsonlines
import time
from argparse import ArgumentParser
import torch
import torch.nn.functional as F

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dict = {}
debug_n = 10000


def to_tensor(x):
    if isinstance(x, list):
        x = torch.tensor(x)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device)


def load_info(args, analysis_setting):
    results_dir = f"{args.base_dir}/activations/{args.model}/{args.task}_{args.uid}/{analysis_setting}"
    info = [None] * debug_n
    with open(f"{results_dir}/record_info.jsonl", "r") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if i >= debug_n:
                break
            info[i] = item
            i += 1
    
    info = info[:i]
    nfiles = len(info)
    print(results_dir, nfiles)
    return info


def check_answer(info_item):
    return info_item['gold_label'] == info_item['pred_label']


def is_same_pred(zsl_item, update_item):
    return zsl_item['pred_label'] == update_item['pred_label']


def normalize_hidden(hidden):
    return F.normalize(hidden, p=2, dim=-1)
 

def prepare_hiddens(infos, mode, key):
    zs_info, icl_info, ftzs_info = infos
    if mode == 'all':
        # check all hidden
        zs_hidden = [info_item[key] for info_item in zs_info]
        icl_hidden = [info_item[key] for info_item in icl_info]
        ftzs_hidden = [info_item[key] for info_item in ftzs_info]
    elif  mode == 'f2t':
        # check only False->True hiddens
        zs_hidden = []
        icl_hidden = []
        ftzs_hidden = []
        n_examples = len(zs_info)
        for i in range(n_examples):
            if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]) and check_answer(icl_info[i]):
                zs_hidden.append(zs_info[i][key])
                icl_hidden.append(icl_info[i][key])
                ftzs_hidden.append(ftzs_info[i][key])

    return zs_hidden, icl_hidden, ftzs_hidden


def analyze_sim(infos, mode, key, normalize=False):
     # n_examples, n_layers, hidden_dim
    zs_hidden, icl_hidden, ftzs_hidden = prepare_hiddens(infos, mode, key)
    
    zs_hidden = to_tensor(zs_hidden)
    icl_hidden = to_tensor(icl_hidden)
    ftzs_hidden = to_tensor(ftzs_hidden)
    
    if normalize:
        zs_hidden = normalize_hidden(zs_hidden)
        icl_hidden = normalize_hidden(icl_hidden)
        ftzs_hidden = normalize_hidden(ftzs_hidden)

    icl_updates = icl_hidden - zs_hidden
    ftzs_updates = ftzs_hidden - zs_hidden

    cos_sim = F.cosine_similarity(icl_updates, ftzs_updates, dim=-1)
    cos_sim = cos_sim.mean(dim=0).cpu().numpy()
    random_updates = to_tensor(np.random.random(ftzs_updates.shape))
    baseline_cos_sim = F.cosine_similarity(icl_updates, random_updates, dim=-1)
    baseline_cos_sim = baseline_cos_sim.mean(dim=0).cpu().numpy()
    return cos_sim, baseline_cos_sim


def pad_attn_map(attn_map, max_len, pad_value):
    padded_map = torch.empty(len(attn_map), len(attn_map[0]), len(attn_map[0][0]), max_len)
    padded_map.fill_(pad_value)
    for i in range(len(attn_map)):
        padded_map[i, :, :, -len(attn_map[i][0][0]):] = torch.Tensor(attn_map[i])
    return padded_map

def analyze_attn_map(infos, mode, key, softmax=False, 
                     sim_func=lambda x, y: F.cosine_similarity(x, y, dim=-1),
                     diff=True):
    # n_examples, n_layers, n_heads, len
    zs_attn_map, icl_attn_map, ftzs_attn_map = prepare_hiddens(infos, mode, key)     
    # pad to max len
    print('pad')
    pad_value = -1e20 if softmax else 0
    max_zs_len = max([len(zs_attn_map[i][0][0]) for i in range(len(zs_attn_map))])
    zs_attn_map = pad_attn_map(zs_attn_map, max_zs_len, pad_value)
    icl_attn_map = pad_attn_map(icl_attn_map, max_zs_len, pad_value)
    ftzs_attn_map = pad_attn_map(ftzs_attn_map, max_zs_len, pad_value)
    
    print('compute')
    if softmax:
        zs_attn_map = F.softmax(zs_attn_map, dim=-1)
        icl_attn_map = F.softmax(icl_attn_map, dim=-1)
        ftzs_attn_map = F.softmax(ftzs_attn_map, dim=-1)
    
    if diff:
        icl_attn_map = icl_attn_map - zs_attn_map
        ft_attn_map = ftzs_attn_map - zs_attn_map

    sim = sim_func(icl_attn_map, ft_attn_map)
    sim = sim.mean(dim=2).mean(dim=0).cpu().numpy()

    return sim


def main(args):
    stt_time = time.time()
    ftzs_info = load_info(args, 'ftzs')
    print(f'loading ftzs data costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    zs_info = load_info(args, 'zs')
    print(f'loading zs data costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    icl_info = load_info(args, 'icl')
    print(f'loading icl data costs {time.time() - stt_time} seconds')
    
    infos = [zs_info, icl_info, ftzs_info]
    # stt_time = time.time()
    # key = 'self_attn_out_hiddens'
    # print('======' * 5, f'analyzing {key}', '======' * 5)
    # cos_sim, baseline_cos_sim = analyze_sim(infos, args.mode, key, normalize=True)
    # results_dict['SimAOU'] = cos_sim.tolist()
    # print("per-layer updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    # cos_sim = cos_sim.mean()
    # print("overall updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    # print()
    # results_dict['Random SimAOU'] = baseline_cos_sim.tolist()
    # print("per-layer updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    # baseline_cos_sim = baseline_cos_sim.mean()
    # print("overall updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    # print()
    # print(f'analyze_sim costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    key = 'attn_map'
    softmax = False
    print('======' * 5, f'analyzing {key}, softmax={softmax}', '======' * 5)
    sim = analyze_attn_map(infos, args.mode, key, softmax=softmax, diff=True)
    results_dict['ICL-FTZS SimAM'] = sim.tolist()
    print("per-layer direct sim (icl)&(zs):\n", np.around(sim, 4))
    mean_sim = sim.mean()
    print("overall direct sim (icl)&(zs):\n", np.around(mean_sim, 4))
    print()
    print(f'analyze_attn_map (w/o softmax) costs {time.time() - stt_time} seconds')


    stt_time = time.time()
    os.makedirs(f'{args.base_dir}/results/{args.model}', exist_ok=True)
    with open(f'{args.base_dir}/results/{args.name}.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f'saving data costs {time.time() - stt_time} seconds')
    stt_time = time.time()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="sst-2")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--uid", type=str, default="0")
    parser.add_argument("--base_dir", type=str, default="artifacts")
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    args.model = f"en_dense_lm_{args.model}"
    main(args)
