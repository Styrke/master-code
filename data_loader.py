import numpy as np

# Example use:
# loader = DataLoader(['data/train/giga-fren.release2'], 32)
# loader.get_batch()
class DataLoader(object):
    def __init__(self, filenames, batch_size, langs=['en','fr'], batch_number=0):
        self.batch_number = batch_number
        self.batch_size = batch_size

        # Read the data and find the length of each line in each language
        lines = dict()
        line_lengths = dict()
        for lang in langs:
            lines[lang] = list()
            for filename in filenames:
                with open(filename + '.' + lang, 'r') as f:
                    lines[lang] += f.read().split('\n')

            line_lengths[lang] = [len(line) for line in lines[lang]]

        # Combined list of lines in both languages
        self.lines = zip(lines[langs[0]], lines[langs[1]])

        # in order to be able to create batches of samples that have approximately the same length, we sort according to length
        self.sortlist = zip(line_lengths['en'],
                       line_lengths['fr'],
                       xrange(len(line_lengths['en'])))
        self.sortlist = sorted(self.sortlist, key=lambda x: 10000*x[0] + x[1])

        self.no_batches = len(self.sortlist)/self.batch_size

        # TODO: we should use a seed or save the permutation so we can reproduce
        self.permutation = np.random.permutation(self.no_batches)

    def get_batch(self):
        batch = self.permutation[self.batch_number]
        start, end = batch*self.batch_size, (batch+1)*self.batch_size
        # NOTE: This may be easier to do with pandas
        batch_line_numbers = [sn for sn in self.sortlist[start:end]]
        self.batch_number += 1
        return batch_line_numbers
