def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    tp = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return round(float(tp / len(y_true)), 4)