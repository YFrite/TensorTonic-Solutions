# Give me numpy pls, i tired boss

def _classification(y_true, y_pred):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1:
            if p == 1: tp += 1
            else: fn += 1
        else:
            if p == 1: fp += 1
            else: tn += 1
            
    n = len(y_true)
    accuracy = (tp + tn) / n
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1_denom = precision + recall
    f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0
    
    return [
        ("accuracy", accuracy),
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
    ]

def _ranking(y_true, y_pred):
    combined = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
    
    rik = sum(item[0] for item in combined[:3])
    total_relevant = sum(y_true)
    
    precision_at_k = rik / 3
    recall_at_k = rik / total_relevant if total_relevant > 0 else 0
    
    return [
        (f"precision_at_3", precision_at_k),
        (f"recall_at_3", recall_at_k),
    ]

def _regression(y_true, y_pred):
    n = len(y_true)

    abs_errors = []
    sq_errors = []
    for t, p in zip(y_true, y_pred):
        diff = p - t
        abs_errors.append(abs(diff))
        sq_errors.append(diff**2)
        
    mae = sum(abs_errors) / n
    rmse = (sum(sq_errors) / n) ** 0.5
    
    return [
        ("mae", mae),
        ("rmse", rmse),
    ]

SYSTEMS = {
    "classification": _classification, 
    "regression": _regression, 
    "ranking": _ranking
}

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    
    return SYSTEMS[system_type](y_true, y_pred)