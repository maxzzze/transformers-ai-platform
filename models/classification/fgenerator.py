import csv
import os
from transformers.data.processors.utils import (
    InputExample,
    DataProcessor
)

csv.field_size_limit(
    2147483647
)  # Increase CSV reader's field limit incase we have long text.


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(
                os.path.join(data_dir, "train.tsv")
            ),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(
                os.path.join(data_dir, "dev.tsv")
            ),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=None,
                    label=label,
                )
            )
        return examples

