python3 run_mt.py \
    --seed 0 \
    --steps_per_task 2e3 \
    --log_every 250 \
    --tasks CW10 \
    --use_popart True \
    --logger_output tsv tensorboard
