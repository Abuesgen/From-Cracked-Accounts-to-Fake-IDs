"""
This module implements a rule based IE pipeline
"""
# pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name,missing-function-docstring,too-many-locals
from typing import Any, Dict, List, Union

import click
import spacy
import srsly
from flair.tokenization import SegtokTokenizer
from spacy.pipeline import EntityRuler
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

from profile_extraction.rule_based_pipeline.spacy_corpus import DoccanoSpacyConverter


class SegtokLowercaseTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.segtok = SegtokTokenizer()

    def __call__(self, text):
        s_tokens = self.segtok.tokenize(text)

        words = [t.text.lower() for t in s_tokens]
        spaces = [t.whitespace_after for t in s_tokens]

        return Doc(self.vocab, words=words, spaces=spaces)


def create_pipeline(pattern_path: str):

    nlp = spacy.blank("de")
    nlp.tokenizer = SegtokLowercaseTokenizer(nlp.vocab)

    nlp.add_pipe("entity_ruler").from_disk(pattern_path)  # type: ignore

    money_ruler: EntityRuler = nlp.add_pipe("entity_ruler", name="money ruler")  # type: ignore

    pattern: List[Dict[str, Union[str, List[Dict[str, Any]]]]] = [{"label": "MONEY", "pattern": [{"LIKE_NUM": True}]}]

    money_ruler.add_patterns(pattern)

    nlp("bitte 2 air max bitte")

    return nlp


@click.command()
@click.argument(
    "pattern_path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument(
    "annotation_path",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
def score_pipeline(pattern_path: str, annotation_path: str):

    nlp = create_pipeline(pattern_path)
    scorer = Scorer(default_pipeline=["entity_ruler"])

    doccano_annotation = list(srsly.read_jsonl(annotation_path))
    dc = DoccanoSpacyConverter(nlp, labels=["PROD", "PAYM", "PER", "MONEY", "LOC", "CRIT"])

    examples = []

    all_spans = 0
    error_spans = 0

    for s in doccano_annotation:
        gold, a, b = dc.convert(s)
        all_spans += a
        error_spans += b
        pred = nlp(s["text"])
        examples += [Example(pred, gold)]

    print(f"{error_spans / all_spans} % wrong spans")

    erg = scorer.score(examples)

    micro_scores = {"p": erg["ents_p"], "r": erg["ents_r"], "f": erg["ents_f"]}
    macro_scores = {"p": 0.0, "r": 0.0, "f": 0.0}

    p_sum = 0.0
    r_sum = 0.0
    f_sum = 0.0

    for label, _ in erg["ents_per_type"].items():
        p_sum += erg["ents_per_type"][label]["p"]
        r_sum += erg["ents_per_type"][label]["r"]
        f_sum += erg["ents_per_type"][label]["f"]

    macro_scores["p"] = p_sum / len(erg["ents_per_type"])
    macro_scores["r"] = r_sum / len(erg["ents_per_type"])
    macro_scores["f"] = f_sum / len(erg["ents_per_type"])

    print(f"MICRO: {micro_scores}")
    print(f"MACRO: {macro_scores}")
    print(erg["ents_per_type"])
