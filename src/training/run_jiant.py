#!/usr/bin/env python3
# -*- coding: utf-8 -*-

EXP_DIR = '/path/to/your/workspace/name_workspace'

import sys
sys.path.insert(0, f"{EXP_DIR}/jiant")

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

#import jiant.scripts.download_data.runscript as downloader
#downloader.download_data(["rte"], f"{EXP_DIR}/tasks")

#snli
#def filter_data(data_row):
    #return data_row.label != -1
    
export_model.export_model(
    hf_pretrained_model_name_or_path="bert-base-uncased", # or /howey/bert-base-uncased-mnli for rte
    output_base_path=f"{EXP_DIR}/models/bert-base-uncased",
)

task_name = "dnli_enriched"

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"{EXP_DIR}/tasks/configs/{task_name}_config.json",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_dir=f"{EXP_DIR}/cache/{task_name}",
    phases=["train", "val","test"],
))

row = caching.ChunkedFilesDataCache(f"{EXP_DIR}/cache/dnli_enriched/train").load_chunk(0)[0]["data_row"]
print(row.input_ids)
print(row.tokens)
print(row.label_id)

row = caching.ChunkedFilesDataCache(f"{EXP_DIR}/cache/dnli_enriched/val").load_chunk(0)[0]["data_row"]
print(row.input_ids)
print(row.tokens)
print(row.label_id)

row = caching.ChunkedFilesDataCache(f"{EXP_DIR}/cache/dnli_enriched/test").load_chunk(0)[0]["data_row"]
print(row.input_ids)
print(row.tokens)
print(row.label_id)

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"{EXP_DIR}/tasks/configs",
    task_cache_base_path=f"{EXP_DIR}/cache",
    train_task_name_list=["dnli_enriched"],
    val_task_name_list=["dnli_enriched"],
    test_task_name_list=["dnli_enriched"],
    train_batch_size=32,
    eval_batch_size=16,
    epochs=1,
    num_gpus=1,
).create_config()
os.makedirs(f"{EXP_DIR}/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, f"{EXP_DIR}/run_configs/run_name_run_config.json")
display.show_json(jiant_run_config)

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=f"{EXP_DIR}/run_configs/run_name_run_config.json",
    output_dir=f"{EXP_DIR}/runs/run_name",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    model_path=f"{EXP_DIR}/models/bert-base-uncased/model/model.p",
    model_config_path=f"{EXP_DIR}/models/bert-base-uncased/model/config.json",
    learning_rate=2e-5,
    eval_every_steps=500,
    #seed=1664796662,
    do_train=True,
    do_val=True,
    do_save=False,
    #no_improvements_for_n_evals=3,
    write_test_preds=True,
    #write_val_preds=True,
    force_overwrite=True,
)
main_runscript.run_loop(run_args)
