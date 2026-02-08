def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    assert len(rater1) == len(rater2)

    n = len(rater1)

    p_0 = sum(rater1[k] == rater2[k] for k in range(n)) / n
    p_e = 0

    for l in set(rater1):
        p_e += (sum(rater1[k] == l for k in range(n)) / n) * (sum(rater2[k] == l for k in range(n)) / n)

    if p_e == 1:
        return 1

    score = (p_0 - p_e) / (1 - p_e)

    return score