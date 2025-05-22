import random
import glob
import sys
from collections import defaultdict

"""
Markov Babbler

After being trained on text from various authors, can
'babble', or generate random walks, and produce text that
vaguely sounds like the author.
"""

class Babbler:
    def __init__(self, n, seed=None):
        self.n = n
        if seed is not None:
            random.seed(seed)

        self.transitions = defaultdict(list)  # maps n-gram tuple to list of next words
        self.starters = []  # list of n-gram tuples that start a sentence
        self.stoppers = []  # list of n-gram tuples that end a sentence

    def add_sentence(self, sentence):
        words = sentence.strip().lower().split()
        if len(words) < self.n:
            return

        # Track starter
        starter = tuple(words[:self.n])
        self.starters.append(starter)

        # Track transitions
        for i in range(len(words) - self.n):
            ngram = tuple(words[i:i + self.n])
            next_word = words[i + self.n]
            self.transitions[ngram].append(next_word)

        # Track stopper
        final_ngram = tuple(words[-self.n:])
        self.transitions[final_ngram].append("EOL")
        self.stoppers.append(final_ngram)

    def add_file(self, filename):
        for line in [line.rstrip().lower() for line in open(filename, errors='ignore').readlines()]:
            self.add_sentence(line)

    def get_starters(self):
        return [' '.join(ngram) for ngram in self.starters]

    def get_stoppers(self):
        return [' '.join(ngram) for ngram in self.stoppers]

    def get_successors(self, ngram):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        return self.transitions.get(ngram, [])

    def get_all_ngrams(self):
        return [' '.join(ngram) for ngram in self.transitions.keys()]

    def has_successor(self, ngram):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        return ngram in self.transitions and len(self.transitions[ngram]) > 0

    def get_random_successor(self, ngram):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        successors = self.get_successors(ngram)
        if not successors:
            return None
        return random.choice(successors)

    def babble(self):
        current_ngram = list(random.choice(self.starters))
        sentence = current_ngram[:]

        while True:
            next_word = self.get_random_successor(tuple(current_ngram))
            if next_word == "EOL" or next_word is None:
                break
            sentence.append(next_word)
            current_ngram = current_ngram[1:] + [next_word]

        return ' '.join(sentence)

def main(n=3, filename='tests/test1.txt', num_sentences=5):
    print(filename)
    babbler = Babbler(n)
    babbler.add_file(filename)

    print(f'num starters {len(babbler.get_starters())}')
    print(f'num ngrams {len(babbler.get_all_ngrams())}')
    print(f'num stoppers {len(babbler.get_stoppers())}')
    for _ in range(num_sentences):
        print(babbler.babble())

if __name__ == '__main__':
    sys.argv.pop(0)
    n = 3
    filename = 'tests/test1.txt'
    num_sentences = 5
    if len(sys.argv) > 0:
        n = int(sys.argv.pop(0))
    if len(sys.argv) > 0:
        filename = sys.argv.pop(0)
    if len(sys.argv) > 0:
        num_sentences = int(sys.argv.pop(0))
    main(n, filename, num_sentences)