from math import exp, log
from collections import Counter
from nltk.util import ngrams


def sentence_bleu(candidate, reference, weights=4):
    """Sentence bleu supporting a single reference."""
    p_ns = []
    # compute modified n-gram precision for each n:
    for n in range(1, weights+1):
        if len(reference) < n:
            continue
        if len(candidate) < n:
            p_ns.append(1)
            continue

        reference_ngrams = ngrams(reference, n)
        reference_counts = Counter(reference_ngrams)

        candidate_ngrams = list(ngrams(candidate, n))
        candidate_counts = Counter(candidate_ngrams)

        hits = 0
        for gram, count in candidate_counts.items():
            if gram in reference_counts:
                hits += min(count, reference_counts[gram])

        # return poor score if no 1-grams match
        if n == 1 and hits == 0:
            return 0

        p_ns.append(hits/len(candidate_ngrams))

    score = exp(sum([1/len(p_ns) * log(p_n) for p_n in p_ns if p_n]))

    # compute brevity penalty (BP)
    if len(candidate) > len(reference):
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1-len(reference)/len(candidate))

    return brevity_penalty * score


if __name__ == '__main__':
    print(sentence_bleu(['this', 'is', 'a', 'test'], ['here', 'is', 'a', 'test']))
