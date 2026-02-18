import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs) == 0:
        return np.array([[]])
        
    lengths = np.fromiter((len(s) for s in seqs), dtype=int)
    
    if max_len is None:
        max_len = lengths.max()
    
    num_samples = len(seqs)

    padded = np.full((num_samples, max_len), pad_value)

    mask = np.arange(max_len) < lengths[:, None]

    all_data = np.concatenate([np.asarray(s)[:max_len] for s in seqs])
    
    padded[mask] = all_data
    
    return padded

