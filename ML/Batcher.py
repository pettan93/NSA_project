import numpy as np
class Batcher:
    """
    Batcher dělí vstupní data na náhodné batche (chunky) dat specifikované velikosti
    """

    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels
        assert len(self.input_data) == len(self.labels)

    def next_batch(self, size):
        indexes = np.arange(len(self.input_data))
        np.random.shuffle(indexes)
        indexes = indexes[:size]
        return ([self.input_data[i] for i in indexes], [self.labels[i] for i in indexes])
