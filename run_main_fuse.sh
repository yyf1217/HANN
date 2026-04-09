export HOM="chordal4_1,g boat,g chordal6,g chordal4_4,v chordal4_1,v chordal5_31,e chordal5_13,e chordal5_24,e"
export ISO="cycle3 cycle4 cycle5 cycle6 chordal4 chordal5"
export LEVEL="g v e"

seed=1
gpus=(0 1 2 3 4 5 6 7)
NGPU=${#gpus[@]}

mkdir -p logs_fuse_5d_2
TASK_FILE="logs_fuse_5d_2/all_tasks.txt"
: > "$TASK_FILE"

# 1. 生成全部任务
for task in $HOM; do
    task_args=$(echo "$task" | tr "," " ")

    # echo "python main_count_add.py --model MP --task $task_args --seed $seed" >> "$TASK_FILE"
    # echo "python main_count_add.py --model Sub --task $task_args --seed $seed" >> "$TASK_FILE"
    echo "python main_count_fuse_5d.py --model L --task $task_args --seed $seed" >> "$TASK_FILE"
    echo "python main_count_fuse_5d.py --model LF --task $task_args --seed $seed" >> "$TASK_FILE"
done

for task in $ISO; do
    for level in $LEVEL; do
        # echo "python main_count_add.py --model MP --task $task $level --seed $seed" >> "$TASK_FILE"
        # echo "python main_count_add.py --model Sub --task $task $level --seed $seed" >> "$TASK_FILE"
        echo "python main_count_fuse_5d.py --model L --task $task $level --seed $seed" >> "$TASK_FILE"
        echo "python main_count_fuse_5d.py --model LF --task $task $level --seed $seed" >> "$TASK_FILE"
    done
done

echo "All tasks written to $TASK_FILE"

# 2. 启动 3 个 worker，分别绑定 GPU 3/4/7，每张卡串行执行自己的任务
for idx in $(seq 0 $((NGPU-1))); do
    gpu=${gpus[$idx]}
    nohup bash -c "
        echo \"[GPU $gpu] worker started at \$(date)\"
        awk '((NR-1) % $NGPU) == $idx' \"$TASK_FILE\" | while IFS= read -r cmd; do
            if [ -n \"\$cmd\" ]; then
                echo \"[GPU $gpu] START: \$cmd\"
                CUDA_VISIBLE_DEVICES=$gpu \$cmd
                echo \"[GPU $gpu] END:   \$cmd\"
            fi
        done
        echo \"[GPU $gpu] worker finished at \$(date)\"
    " > "logs_fuse_5d_2/gpu_${gpu}.log" 2>&1 &
done

echo "All GPU workers submitted."