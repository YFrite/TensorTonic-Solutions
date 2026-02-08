def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    assert len(rater1) == len(rater2)

    n = len(rater1)

    p_0 = sum(rater1[k] == rater2[k] for k in range(n)) / n
    p_e = 0

    labels = set(rater1)
    counts1 = {l: rater1.count(l) for l in labels}
    counts2 = {l: rater2.count(l) for l in labels}

    for l in labels:
        p_e += (counts1[l] / n) * (counts2[l] / n)

    if p_e == 1:
        return 1

    score = (p_0 - p_e) / (1 - p_e)

    return score