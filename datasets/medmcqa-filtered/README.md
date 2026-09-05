---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: case_prompt
    dtype: string
  - name: diagnostic_reasoning
    dtype: string
  - name: final_diagnosis
    dtype: string
  splits:
  - name: train
    num_bytes: 1950642
    num_examples: 1347
  download_size: 1127020
  dataset_size: 1950642
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
