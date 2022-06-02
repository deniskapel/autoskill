from globals import ID2Midas
import random

class MidasSampler:

    def __init__(self):
        self.bins = {label: 0 for label in ID2Midas}

    def balanced_sample(self, samples, bin_size=100) -> dict:
        """reduce the number of large classes"""
        output = list()
        random.Random(42).shuffle(samples)
        for sample in samples:
            label = sample['predict']['midas']

            if self.bins[label] >= bin_size:
                continue

            output.append(sample)
            self.bins[label] += 1

        return output
