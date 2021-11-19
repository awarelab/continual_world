TASK_SEQS = {
    "CW10": [
        "hammer-v1",
        "push-wall-v1",
        "faucet-close-v1",
        "push-back-v1",
        "stick-pull-v1",
        "handle-press-side-v1",
        "push-v1",
        "shelf-place-v1",
        "window-close-v1",
        "peg-unplug-side-v1",
    ],
}

TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
