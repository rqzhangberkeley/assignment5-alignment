for num_prompts_per_step in 512 1024 2048; do
  for train_batch_size in 32 128; do
    for lr in 1e-5 3e-5 1e-4; do
      run_name="expert_iteration_lr_${lr}_B_${num_prompts_per_step}_Btrain_${train_batch_size}_wd_0.0"
      log_path="./logs/${run_name}.log"
      nohup uv run python ./cs336_alignment/expert_iteration.py \
        --run_name "${run_name}" \
        --num_prompts_per_step "${num_prompts_per_step}" \
        --train_batch_size "${train_batch_size}" \
        --lr "${lr}" \
        > "${log_path}" 2>&1 &
      wait
      sleep 30
    done
  done
done