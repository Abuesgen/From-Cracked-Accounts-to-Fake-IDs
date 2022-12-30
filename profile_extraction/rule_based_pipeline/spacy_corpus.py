"""
This module represents a spacy corpus
"""
# pylint: disable=too-few-public-methods,missing-class-docstring,too-many-locals

import random
import re

from spacy.tokens import Doc, DocBin


class NoValidEntitySpanError(Exception):
    """Raised when the input value is too large"""


class CharBasedEntity:
    """
    Represents a char based label
    """

    def __init__(self, start_char, end_char, label, text=None):
        self.start_char = start_char
        self.end_char = end_char
        self.label = label
        self.text = text[start_char:end_char]

    def __repr__(self):
        return f"({self.start_char}, {self.end_char})"


class DoccanoSample:
    """
    This class represents a single doccano entry
    """

    def __init__(self, jsonl: dict):
        self.text = jsonl["data"]
        self.entities = [
            CharBasedEntity(start_char=e[0], end_char=e[1], label=e[2], text=jsonl["data"]) for e in jsonl["label"]
        ]


class DoccanoSpacyConverter:
    def __init__(self, nlp, labels=None):
        self.nlp = nlp
        self.labels = labels or []

    def __call__(self, *args, **kwargs):
        return self.convert(*args, **kwargs)

    def convert(self, doccano):
        """
        Converts doccano DataPoints to Spacy docs
        :param doccano: doccano datapoint
        :return:spacy doc
        """
        doc: Doc = self.nlp(doccano["text"])

        entities = []

        all_spans = 0
        error_spans = 0

        for char_ent_annotation in doccano["entities"]:

            if self.labels and char_ent_annotation[2] not in self.labels:
                continue

            span = doc.char_span(
                char_ent_annotation[0],
                char_ent_annotation[1],
                alignment_mode="contract",
                label=char_ent_annotation[2],
            )

            all_spans += 1
            if span is None:
                text = doccano["text"].lower()
                ent_text = text[char_ent_annotation[0] : char_ent_annotation[1]]
                search_start = max(0, char_ent_annotation[0] - 25)
                search_end = min(len(doc.text), char_ent_annotation[1] + 25)
                entity_match = re.search(ent_text, doc.text[search_start:search_end])
                if entity_match:
                    ent_start, ent_end = entity_match.start() + search_start, entity_match.end() + search_start

                    span = doc.char_span(ent_start, ent_end, alignment_mode="contract", label=char_ent_annotation[2])

            if span is None:

                tokens = [
                    f"'{t}'" for t in doc if char_ent_annotation[0] - 20 <= t.idx <= char_ent_annotation[1] + 20
                ]

                print(f"Could not set entity {ent_text} with current tokenization {tokens}")
                error_spans += 1

                continue

            entities += [span]

        if entities:
            try:
                doc.set_ents(entities)
            except:  # pylint: disable=bare-except
                pass

        return doc, all_spans, error_spans


def create_spacy_doc(data):
    """
    Creates spacy docs from raw data
    :param data: raw doccano sample
    :return:a spacy doc
    """
    converter = DoccanoSpacyConverter()  # pylint: disable=no-value-for-parameter

    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    docs = []
    correct_spans = 0
    error_spans = 0

    for sample in data:
        doccano_sample = DoccanoSample(sample)

        doc, correct, error = converter(doccano_sample)

        correct_spans += correct
        error_spans += error

        docs += [doc]

    print(
        f"{error_spans} / {correct_spans} = {error_spans / correct_spans} could not be converted to spacy entities."
    )

    random.shuffle(docs)

    num_train_samples = int(0.8 * len(docs))

    train_docs = docs[:num_train_samples]

    dev_docs = docs[num_train_samples:]

    train_doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    dev_doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    for doc in train_docs:
        train_doc_bin.add(doc)

    for doc in dev_docs:
        dev_doc_bin.add(doc)

    train_doc_bin.to_disk("_train.spacy")
    dev_doc_bin.to_disk("_dev.spacy")

    return doc_bin
