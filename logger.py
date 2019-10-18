"""Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514."""
import os

import tensorflow as tf


class Logger(object):
    """Tensorboard logger class."""
    def __init__(self, log_dir: str, name: str = None):
        """Create a summary writer logging to log_dir.

        Args:
            log_dir: Directory where tensorboard logs are to be saved.
            name: Name of the sub-folder.

        """
        if name is None:
            name = "temp"
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            self.writer = tf.summary.create_file_writer(os.path.join(
                log_dir, name),
                                                        filename_suffix=name)
        else:
            self.writer = tf.summary.create_file_writer(log_dir,
                                                        filename_suffix=name)

    def scalar_summary(self, tag: str, value: float, step: int):
        """Log a scalar variable.

        Args:
            tag: Tag for the variable being logged.
            value: Value of the variable.
            step: Iteration step number.

        """
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
