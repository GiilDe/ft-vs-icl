import json
import os
import numpy as np
import sys
import copy
import jsonlines
import time
import scipy.stats

task = sys.argv[1]
mode = sys.argv[2]
model = sys.argv[3]
model = f"en_dense_lm_{model}"
uid = sys.argv[4]

# !!! replace by your $base_dir/ana_rlt here
ana_dir = "base_dir/ana_rlt"
ana_model_dir = f"{ana_dir}/{model}"

save_rlts = {}
debug_scale = 1
debug_n = 10000


def load_info(uid, ana_setting):
    rlt_dir = f"{ana_model_dir}/{task}_{uid}/{ana_setting}"
    info = [None] * debug_n
    with open(f"{rlt_dir}/record_info.jsonl", "r") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if i >= debug_n:
                break
            info[i] = item
            i += 1
    
    info = info[:i]
    nfiles = len(info)
    print(rlt_dir, nfiles)
    info = info[:nfiles//debug_scale]
    return info


def calc_cos_sim(v1, v2):
    num = (v1 * v2).sum(axis=-1)  # dot product
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-20  # length
    res = num / denom
    return res


def np_softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def check_answer(info_item):
    return info_item['gold_label'] == info_item['pred_label']


def is_same_pred(zsl_item, update_item):
    return zsl_item['pred_label'] == update_item['pred_label']


def normalize_hidden(hidden):
    norm = np.linalg.norm(hidden, axis=-1) + 1e-20  # length
    norm = norm[:, :, np.newaxis]
    hidden = hidden / norm
    return hidden


def prepare_hiddens(infos, mode, key, normalize=False):
    zs_info, icl_info, ftzs_info = infos
    if mode == 'all':
        # ====================  check all hidden ======================
        zs_hidden = [info_item[key] for info_item in zs_info]
        icl_hidden = [info_item[key] for info_item in icl_info]
        ftzs_hidden = [info_item[key] for info_item in ftzs_info]
    elif  mode == 'f2t':
        # ================== check False->True hidden =================
        zs_hidden = []
        icl_hidden = []
        ftzs_hidden = []
        n_examples = len(zs_info)
        for i in range(n_examples):
            if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]) and check_answer(icl_info[i]):
                zs_hidden.append(zs_info[i][key])
                icl_hidden.append(icl_info[i][key])
                ftzs_hidden.append(ftzs_info[i][key])

    if normalize:
        zs_hidden = normalize_hidden(zs_hidden)
        icl_hidden = normalize_hidden(icl_hidden)
        ftzs_hidden = normalize_hidden(ftzs_hidden)

    return zs_hidden, icl_hidden, ftzs_hidden


def analyze_sim(infos, mode, key, normalize=False):
     # n_examples, n_layers, hidden_dim
    zs_hidden, icl_hidden, ftzs_hidden = prepare_hiddens(
        infos, mode, key, normalize=normalize
    )
    zs_hidden = np.array(zs_hidden)
    icl_hidden = np.array(icl_hidden)
    ftzs_hidden = np.array(ftzs_hidden)

    icl_updates = icl_hidden - zs_hidden
    ftzs_updates = ftzs_hidden - zs_hidden

    print('======' * 5, f'analyzing {key}', '======' * 5)

    cos_sim = calc_cos_sim(icl_updates, ftzs_updates)
    cos_sim = cos_sim.mean(axis=0)
    save_rlts['SimAOU'] = cos_sim.tolist()
    print("per-layer updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    cos_sim = cos_sim.mean()
    print("overall updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    print()

    random_updates = np.random.random(ftzs_updates.shape)
    baseline_cos_sim = calc_cos_sim(icl_updates, random_updates)
    baseline_cos_sim = baseline_cos_sim.mean(axis=0)
    save_rlts['Random SimAOU'] = baseline_cos_sim.tolist()
    print("per-layer updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    baseline_cos_sim = baseline_cos_sim.mean()
    print("overall updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    print()


def analyze_attn_map(infos, mode, key, softmax=False, sim_func=calc_cos_sim, diff=True):
    # n_examples, n_layers, n_heads, len
    zs_attn_map, icl_attn_map, ftzs_attn_map = prepare_hiddens(
        infos, mode, key, normalize=False
    )     
    # pad to max len
    pad_value = -1e20 if softmax else 0
    max_zs_len = max([len(zs_attn_map[i][0][0]) for i in range(len(zs_attn_map))])
    zs_attn_map = np.array([[[[pad_value] * (max_zs_len - len(head)) + head for head in layer] for layer in example] for example in zs_attn_map])
    icl_attn_map = np.array([[[[pad_value] * (max_zs_len - len(head)) + head for head in layer] for layer in example] for example in icl_attn_map])
    ftzs_attn_map = np.array([[[[pad_value] * (max_zs_len - len(head)) + head for head in layer] for layer in example] for example in ftzs_attn_map])
    
    if softmax:
        zs_attn_map = np_softmax(zs_attn_map, axis=-1)
        icl_attn_map = np_softmax(icl_attn_map, axis=-1)
        ftzs_attn_map = np_softmax(ftzs_attn_map, axis=-1)
    
    print('======' * 5, f'analyzing {key}, softmax={softmax}', '======' * 5)
    if diff:
        icl_attn_map_update = icl_attn_map - zs_attn_map
        ft_attn_map_update = ftzs_attn_map - zs_attn_map
        sim = sim_func(icl_attn_map_update, ft_attn_map_update)
        sim = sim.mean(axis=2).mean(axis=0)
        save_rlts['ICL-FTZS SimAM'] = sim.tolist()
        print("per-layer direct sim (icl)&(zs):\n", np.around(sim, 4))
        mean_sim = sim.mean()
        print("overall direct sim (icl)&(zs):\n", np.around(mean_sim, 4))
        print()
    else:
        sim = sim_func(icl_attn_map, zs_attn_map)
        sim = sim.mean(axis=2).mean(axis=0)
        save_rlts['ZSL SimAM'] = sim.tolist()
        print("per-layer direct sim (icl)&(zs):\n", np.around(sim, 4))
        mean_sim = sim.mean()
        print("overall direct sim (icl)&(zs):\n", np.around(mean_sim, 4))
        print()

        sim = sim_func(icl_attn_map, ftzs_attn_map).mean(axis=2).mean(axis=0)
        save_rlts['SimAM'] = sim.tolist()
        print("per-layer direct sim (icl)&(ftzs):\n", np.around(sim, 4))
        mean_sim = sim.mean()
        print("overall direct sim (icl)&(ftzs):\n", np.around(mean_sim, 4))
        print()


def main():
    stt_time = time.time()
    ftzs_info = load_info(uid, 'ftzs')
    print(f'loading ftzs data costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    zs_info = load_info(uid, 'zs')
    print(f'loading zs data costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    icl_info = load_info(uid, 'icl')
    print(f'loading icl data costs {time.time() - stt_time} seconds')
    infos = [zs_info, icl_info, ftzs_info]
    stt_time = time.time()
    analyze_sim(infos, mode, 'self_attn_out_hiddens', normalize=True)
    print(f'analyze_sim costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    analyze_attn_map(infos, mode, 'attn_map', softmax=False, 
                     sim_func=calc_cos_sim, diff=True)
    print(f'analyze_attn_map (w/o softmax) costs {time.time() - stt_time} seconds')
    
    stt_time = time.time()
    with open(f'{ana_dir}/rlt_json/{uid}-{task}-{model}.json', 'w') as f:
        json.dump(save_rlts, f, indent=2)
    print(f'saving data costs {time.time() - stt_time} seconds')
    stt_time = time.time()

if __name__ == "__main__":
    main()
