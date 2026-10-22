import datetime
import time


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: time as a float
    :return: time as a formatted string
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def report_progress(step, n_steps, t0, print_every=40, step_str="Step", eval_str="Eval", eval_val=None):
    # Progress update every `print_every` steps.
    if step % print_every == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        # Report progress.
        progress_str = "  {:} {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step_str, step, n_steps, elapsed)
        if eval_val is not None:
            progress_str += " {:}: {:>5,}.".format(eval_str, eval_val)
        print(progress_str)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
