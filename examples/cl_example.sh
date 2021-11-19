python3 run_cl.py \
    --seed 0 \
    --steps_per_task 2e3 \
    --log_every 250 \
    --tasks CW20 \
    --cl_method ewc \
    --cl_reg_coef 1e4 \
    --logger_output tsv tensorboard
