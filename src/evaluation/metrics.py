import numpy as np
from numpy.linalg import norm


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def stability(embeddings):
    """
    Measures how consistent outputs are within a language.
    Returns value in [0, 1]
    """
    sims = [
        cosine(x, y)
        for x in embeddings
        for y in embeddings
    ]
    return np.mean(sims)


def bounded_intent_drift(*language_outputs, embed_fn):
    """
    Returns a drift score in [0, 1]

    0 = identical behavior
    1 = maximum divergence
    """

    # 1. Embed all outputs
    embeddings = [embed_fn(outputs) for outputs in language_outputs]

    # 2. Cross-language distance
    pairwise_dist = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims = [
                cosine(x, y)
                for x in embeddings[i]
                for y in embeddings[j]
            ]
            pairwise_dist.append(1 - np.mean(sims))  # distance

    between = np.mean(pairwise_dist)

    # 3. Within-language instability
    within = np.mean([
        1 - stability(e)
        for e in embeddings
    ])

    # 4. Combine and clamp
    drift = (between + within) / 2

    return float(np.clip(drift, 0, 1))