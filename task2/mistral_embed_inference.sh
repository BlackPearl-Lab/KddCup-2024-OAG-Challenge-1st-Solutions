#!/bin/bash

# 启动8个并行Python进程
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python mistral_embedding.py --gpu $i  &
done

# 等待所有后台进程完成
wait


echo "所有Python程序已完成运行。"