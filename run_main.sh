#!/bin/bash

export HOM="chordal4_1,g boat,g chordal6,g chordal4_4,v chordal4_1,v chordal5_31,e chordal5_13,e chordal5_24,e"
export ISO="cycle3 cycle4 cycle5 cycle6 chordal4 chordal5"
export LEVEL="g v e"

seed=1
NGPU=8

mkdir -p logs_add
TASK_FILE="logs_add/all_tasks.txt"
: > "$TASK_FILE"

# 1. 生成全部任务
for task in $HOM; do
    task_args=$(echo "$task" | tr "," " ")

    echo "python main_count_add.py --model MP --task $task_args --seed $seed" >> "$TASK_FILE"
    echo "python main_count_add.py --model Sub --task $task_args --seed $seed" >> "$TASK_FILE"
    echo "python main_count_add.py --model L --task $task_args --seed $seed" >> "$TASK_FILE"
    echo "python main_count_add.py --model LF --task $task_args --seed $seed" >> "$TASK_FILE"
done

for task in $ISO; do
    for level in $LEVEL; do
        echo "python main_count_add.py --model MP --task $task $level --seed $seed" >> "$TASK_FILE"
        echo "python main_count_add.py --model Sub --task $task $level --seed $seed" >> "$TASK_FILE"
        echo "python main_count_add.py --model L --task $task $level --seed $seed" >> "$TASK_FILE"
        echo "python main_count_add.py --model LF --task $task $level --seed $seed" >> "$TASK_FILE"
    done
done

echo "All tasks written to $TASK_FILE"

# 2. 启动 8 个 worker，每个 worker 只占用 1 张 GPU，串行执行自己的任务
for gpu in $(seq 0 $((NGPU-1))); do
    nohup bash -c "
        echo \"[GPU $gpu] worker started at \$(date)\"
        awk '((NR-1) % $NGPU) == $gpu' \"$TASK_FILE\" | while IFS= read -r cmd; do
            if [ -n \"\$cmd\" ]; then
                echo \"[GPU $gpu] START: \$cmd\"
                CUDA_VISIBLE_DEVICES=$gpu \$cmd
                echo \"[GPU $gpu] END:   \$cmd\"
            fi
        done
        echo \"[GPU $gpu] worker finished at \$(date)\"
    " > "logs_add/gpu_${gpu}.log" 2>&1 &
done

echo "All GPU workers submitted."


# CUDA_VISIBLE_DEVICES=1 nohup python main_count_add.py --model Sub --task chordal5 v --seed 1 >debug.txt 2>&1 &