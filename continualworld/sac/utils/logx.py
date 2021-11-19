"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import atexit
import json
import os
import os.path as osp
import time

import numpy as np
import tensorflow as tf

from continualworld.sac.utils.serialization_utils import convert_json
from continualworld.utils.utils import get_random_string, get_readable_timestamp

color2num = dict(
    gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
        self,
        logger_output,
        config,
        group_id,
        output_dir=None,
        output_fname="progress.tsv",
        exp_name=None,
        with_mrunner=False,
    ):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``./experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.logger_output = logger_output

        run_id = get_readable_timestamp() + "_" + get_random_string()
        self.output_dir = output_dir or f"./logs/{group_id}/{run_id}"
        if osp.exists(self.output_dir):
            print(f"Warning: Log dir {self.output_dir} already exists! Storing info there anyway.")
        else:
            os.makedirs(self.output_dir)

        if "tsv" in self.logger_output:
            self.output_file = open(osp.join(self.output_dir, output_fname), "w")
            atexit.register(self.output_file.close)

        if "neptune" in self.logger_output:
            if with_mrunner:
                import mrunner

                self._neptune_exp = mrunner.helpers.client_helper.experiment_
            else:
                import neptune

                neptune.init()  # env variable NEPTUNE_PROJECT is used
                self._neptune_exp = neptune.create_experiment()

        if "tensorboard" in self.logger_output:
            self.tb_writer = tf.summary.create_file_writer(self.output_dir)
            self.tb_writer.set_as_default()

        self.save_config(config)

        print(colorize(f"Logging data to {self.output_dir}", "green", bold=True))

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
        print(colorize("Saving config:\n", color="cyan", bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            out.write(output)

    def setup_tf_saver(self, sess, inputs, outputs):
        """
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {
            "inputs": {k: v.name for k, v in inputs.items()},
            "outputs": {k: v.name for k, v in outputs.items()},
        }

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        step = self.log_current_row.get("total_env_steps")
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)

            # Log to Neptune
            if "neptune" in self.logger_output:
                # Try several times.
                for _ in range(10):
                    try:
                        self._neptune_exp.send_metric(key, step, val)
                    except:
                        time.sleep(5)
                    else:
                        break
            if "tensorboard" in self.logger_output:
                tf.summary.scalar(key, data=val, step=step)

        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, d):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in d.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            stats = self.get_stats(key)
            super().log_tabular(key if average_only else key + "/avg", stats[0])
            if not (average_only):
                super().log_tabular(key + "/std", stats[1])
            if with_min_and_max:
                super().log_tabular(key + "/max", stats[3])
                super().log_tabular(key + "/min", stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict.get(key)
        if not v:
            return [None, None, None, None]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return [np.mean(vals), np.std(vals), np.min(vals), np.max(vals)]
