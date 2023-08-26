#!/usr/bin/python3
set -ex


# =================== analyze 125m GPT ===================

model=125m

task=cb
python3 icl_ft/compute_sim.py $task all $model

task=sst2
python3 icl_ft/compute_sim.py $task all $model

task=sst5
python3 icl_ft/compute_sim.py $task all $model

task=subj
python3 icl_ft/compute_sim.py $task all $model

task=mr
python3 icl_ft/compute_sim.py $task all $model

task=agnews
python3 icl_ft/compute_sim.py $task all $model

# # =================== analyze 1.3b GPT ===================

model=1_3b

task=cb
python3 icl_ft/compute_sim.py $task all $model

task=sst2
python3 icl_ft/compute_sim.py $task all $model

task=sst5
python3 icl_ft/compute_sim.py $task all $model

task=subj
python3 icl_ft/compute_sim.py $task all $model

task=mr
python3 icl_ft/compute_sim.py $task all $model

task=agnews
python3 icl_ft/compute_sim.py $task all $model

# =================== analyze 2.7b GPT ===================

model=2_7b

task=cb
python3 icl_ft/compute_sim.py $task all $model

task=sst2
python3 icl_ft/compute_sim.py $task all $model

task=sst5
python3 icl_ft/compute_sim.py $task all $model

task=subj
python3 icl_ft/compute_sim.py $task all $model

task=mr
python3 icl_ft/compute_sim.py $task all $model

task=agnews
python3 icl_ft/compute_sim.py $task all $model