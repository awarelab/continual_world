from input_args import cl_parse_args, mt_parse_args, single_parse_args

PARSE_ARGS_DICT = {"cl": cl_parse_args, "mt": mt_parse_args, "single": single_parse_args}


def combine_config_with_defaults(config):
    run_kind = config["run_kind"]
    res = vars(PARSE_ARGS_DICT[run_kind]([]))  # get defaults
    res.update(config)
    return res
