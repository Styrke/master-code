from math import exp, log
from collections import Counter
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from subprocess import Popen, PIPE
import subprocess

def moses_bleu(translated, reference):
    """Using moses bleu implementation multi-bleu.perl"""
    call = ["%s %s %s %s %s" %("perl utils/multi-bleu.perl", reference, " < ", translated, "| awk '{ print $3 }' | sed 's/,//'")]
    call1 = 'perl utils/multi-bleu.perl ' + reference + ' < ' + translated
    subprocess.call(call1, shell=True)
    p = Popen(call, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    return float(out)

def corpus_bleu(candidates, references, max_n=4):
    """Corpus bleu supporting a single reference per candidate."""
    p_ns = []
    # compute modified n-gram precision for each n:
    for n in range(1, max_n+1):
        count_clip = 0
        count = 0
        for candidate, reference in zip(candidates, references):
            if len(reference) < n or len(candidate) < n:
                continue

            reference_ngrams = ngrams(reference, n)
            reference_counts = Counter(reference_ngrams)

            candidate_ngrams = list(ngrams(candidate, n))
            candidate_counts = Counter(candidate_ngrams)

            for gram, cnt in candidate_counts.items():
                if gram in reference_counts:
                    count_clip += min(cnt, reference_counts[gram])

            count += len(candidate_ngrams)

        # avoid returning p_n = 0 because log(0) is undefined
        if count_clip == 0:
            if n == 1:
                return 0
            else:
                count_clip = 1

        if count:
            p_ns.append(count_clip/count)

    score = exp(sum([1/len(p_ns) * log(p_n) for p_n in p_ns]))

    # compute brevity penalty (BP)
    c = r = 0
    for candidate, reference in zip(candidates, references):
        c += min(len(candidate), len(reference))
        r += len(reference)
    brevity_penalty = exp(1-r/c)

    return brevity_penalty * score


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

        # avoid returning p_n = 0 because log(0) is undefined
        if hits == 0:
            if n == 1:
                return 0
            else:
                hits = 1

        p_ns.append(hits/len(candidate_ngrams))

    score = exp(sum([1/len(p_ns) * log(p_n) for p_n in p_ns]))

    # compute brevity penalty (BP)
    if len(candidate) > len(reference):
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1-len(reference)/len(candidate))

    return brevity_penalty * score


def mean_char_edit_distance(candidates, references):
    total_distance = 0
    total_target_length = 0
    for y, t in zip(candidates, references):
        total_distance += edit_distance(y, t)
        total_target_length += len(t)
    return total_distance/total_target_length

if __name__ == '__main__':
    candidate = ['this', 'is', 'a', 'test']
    reference = ['here', 'is', 'a', 'test']
    candidate2 = ['this', 'is', 'another', 'test']
    reference2 = ['this', 'is', 'another', 'test']
    print(corpus_bleu([candidate, candidate2], [reference, reference2]))
    print(sentence_bleu(candidate, reference))
