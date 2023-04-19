from math import inf
from dataclasses import dataclass
import numpy


@dataclass
class ActivationMetrics:
    # SHOULD be activated and predicted as activated
    true: float = inf
    # SHOULD NOT be activated but predicted activated
    false: float = inf
    # common activation rate
    rate: float = inf


@dataclass
class Metrics:
    p1: float = inf
    p5: float = inf
    median: float = inf
    p95: float = inf
    p99: float = inf
    avg: float = inf
    activations: ActivationMetrics = ActivationMetrics()
    wer: float = inf
    ser: float = inf


def calculate_activations_metrics(preds, labels):
    # SHOULD be activated and predicted as activated
    true_activations = (labels > 0) * (preds > 0)
    # SHOULD NOT be activated but predicted activated
    false_activations = (labels == 0) * (preds > 0)
    # metrics
    true_activations = true_activations.sum() / (preds > 0).sum()
    false_activations = false_activations.sum() / (preds > 0).sum()
    rate = (labels > 0).sum() / len(labels)
    return ActivationMetrics(true_activations, false_activations, rate)


def calculate_wer(eou_timing: float, intervals: list):
    if intervals is None or not len(intervals):
        return None
    deletions = 0
    for _, end in reversed(intervals):
        if eou_timing >= end:
            break
        else:
            deletions += 1
    return deletions / len(intervals)


def calculate_ser(eou_timing: float, intervals: list):
    if intervals is None or not len(intervals):
        return None
    _, phrase_end = intervals[-1]
    return float(eou_timing < phrase_end)


def create_server_predictions(
    probs,
    eou_ml_threshold,
    eou_ml_window_threshold,
    eou_absolute_ml_threshold=0.98,
    eou_fall_tollerance=0.04,
    eou_fall_tollerance_min=0.90,
    **__,
):
    def server_eou(probs):
        # abs threshold
        abs_activation = probs > eou_absolute_ml_threshold
        abs_activation_indexes = numpy.where(abs_activation)[0]
        if abs_activation_indexes.size != 0:
            return abs_activation_indexes[0]
        # decrease
        fall_min_mask = probs > eou_fall_tollerance_min
        diff = numpy.diff(probs)
        diff = numpy.concatenate((numpy.zeros((1,)), diff))
        diff = diff * fall_min_mask
        fall_mask = ~(diff <= -eou_fall_tollerance)
        # threshold
        activated = probs > eou_ml_threshold
        activated = activated * fall_mask
        # find sequence of activated
        activated_p = numpy.concatenate(([0], activated.view(numpy.int8), [0]))
        transitions = numpy.abs(numpy.diff(activated_p))
        intevals = numpy.where(transitions)[0].reshape(-1, 2)
        lens = numpy.diff(intevals).reshape(-1)
        lens_mask = lens > eou_ml_window_threshold
        lens = lens * lens_mask
        lens_activated = lens.nonzero()[0]
        intevals_activated = intevals[lens_activated]
        if intevals_activated.size != 0:
            # take first interval, its start position
            # and add minimum windows number to be activated
            return intevals_activated[0][0] + eou_ml_window_threshold - 1
        else:
            # not activated
            return probs.shape[-1]

    if isinstance(probs, numpy.ndarray):
        preds = numpy.apply_along_axis(server_eou, axis=1, arr=probs)
    else:
        preds = [server_eou(probs_item) for probs_item in probs]
        preds = numpy.array(preds)
    return preds


def calculate_metrics(preds: numpy.ndarray, labels: numpy.ndarray, intervals: list, eou_step_seconds: float) -> Metrics:
    eou_pred = preds * eou_step_seconds
    diff = preds - labels
    p1 = numpy.percentile(diff, 1) * eou_step_seconds
    p5 = numpy.percentile(diff, 5) * eou_step_seconds
    median = numpy.percentile(diff, 50) * eou_step_seconds
    p95 = numpy.percentile(diff, 95) * eou_step_seconds
    p99 = numpy.percentile(diff, 99) * eou_step_seconds
    avg = numpy.mean(diff) * eou_step_seconds
    # activations
    activations = calculate_activations_metrics(preds, labels)
    # WER & SER
    wer = numpy.array([calculate_wer(*args) for args in zip(eou_pred, intervals)])
    ser = numpy.array([calculate_ser(*args) for args in zip(eou_pred, intervals)])
    wer = wer[wer != numpy.array(None)]
    ser = ser[ser != numpy.array(None)]
    if len(wer):
        wer = wer.mean()
    else:
        wer = 1.0
    if len(ser):
        ser = ser.mean()
    else:
        ser = 1.0
    return Metrics(p1, p5, median, p95, p99, avg, activations, wer, ser)


def create_metrics_message(phase, epoch_loss, metrics: Metrics, print_loss=True):
    if print_loss:
        loss_message = f'Loss: {epoch_loss:.4f}'
    else:
        loss_message = ''
    message = (
        f'{phase} {loss_message} \n'
        + f'p1: {metrics.p1:.4f} p5: {metrics.p5:.4f} '
        + f'p50: {metrics.median:.4f} p95: {metrics.p95:.4f} p99: {metrics.p99:.4f} '
        + f'avg: {metrics.avg:1.4f}\n'
        + f'WER: {(metrics.wer * 100.0):.2f}% '
        + f'SER: {(metrics.ser * 100.0):.2f}%\n'
        + f'True  Activations: {(metrics.activations.true * 100):.1f}% '
        + f'(Should be activated and predicted activated)\n'
        + f'False Activations: {(metrics.activations.false * 100):.1f}% '
        + f'(Should NOT be activated but predicted activated)\n'
        + f'Activation Rate: {(metrics.activations.rate * 100):.1f}%\n'
    )
    return message
