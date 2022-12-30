"""
This module contains functions for embedding Profiles
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from flair.data import Sentence
from flair.embeddings import (
    FlairEmbeddings,
    TokenEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.tokenization import SegtokTokenizer
from numpy.typing import NDArray
from tqdm import tqdm

from profile_extraction.ner_model.train_fasttext import LowercaseFastTextEmbeddings
from profile_extraction.profile_creation.profile import (
    ProductComponent,
    Profile,
    ProfileCollection,
)

log = logging.getLogger(__name__)


class ProfileEmbedder(ABC):
    """
    Creates a vector representation for a profile
    """

    LOWERCASED_FASTTEXT = "LOWERCASED_FASTTEXT"
    MODEL = "MODEL"
    WORD = "WORD"
    FLAIR = "FLAIR"
    TRANSFORMER = "TRANSFORMER"

    def __init__(self, embedder: TokenEmbeddings):
        self.embedder = embedder

    @abstractmethod
    def embed(
        self, profiles: ProfileCollection
    ) -> Tuple[Tuple[List[Profile], List[NDArray]], Tuple[List[Profile], List[NDArray]]]:
        """
        Embeds a whole ProfileCollection
        :param profiles: Profiles to embed
        :return: A tuple containing (profiles with nonzero vectors, vectors),(profiles with zero vectors, vectors)
        """

    @abstractmethod
    def embed_single(self, profile: Profile, **kwargs) -> NDArray:
        """
        Computes a vector representing a profile

        :param profile: Profile to compute the vector for
        :return: Vector representing a profile
        """


class ProfileTransformerEmbedder(ProfileEmbedder):
    """
    Create Profile Embeddings using contextualized Transformer Embeddings
    """

    def __init__(self, model: SequenceTagger):
        self._model = model
        super().__init__(model.embeddings)

    def embed_single(self, profile: Profile, **kwargs) -> NDArray:
        product_components = [product for product in profile.components if isinstance(product, ProductComponent)]
        product_messages = list({product.message for product in product_components})
        message_strs = [
            message.message
            for message in product_messages
            if message.message is not None and len(message.message.strip()) > 0
        ]
        message_sentences = [Sentence(message, use_tokenizer=SegtokTokenizer()) for message in message_strs]

        self._model.predict(message_sentences, embedding_storage_mode="cpu", mini_batch_size=16)

        product_embeddings = []
        for sentence in tqdm(message_sentences, desc="Embedding Products", position=1, leave=False):
            for entity_span in sentence.get_spans("ner"):
                if entity_span.labels[0].value == "PROD":
                    token_embeddings = []
                    for token in entity_span:
                        token_embeddings.append(token.embedding.detach().cpu().numpy())
                    product_embeddings.append(np.mean(np.column_stack(token_embeddings), axis=1))

        profile_embedding = np.zeros(self._model.embeddings.embedding_length)
        if len(product_embeddings) > 0:
            profile_embedding = np.array(np.mean(np.column_stack(product_embeddings), axis=1))

        if profile_embedding.shape != (self._model.embeddings.embedding_length,):
            raise RuntimeError("The created vector does not have the expected length.")

        return profile_embedding

    def embed(
        self, profiles: ProfileCollection
    ) -> Tuple[Tuple[List[Profile], List[NDArray]], Tuple[List[Profile], List[NDArray]]]:

        vectors = [self.embed_single(profile) for profile in tqdm(profiles, desc="Embedding Profiles")]

        zero_profiles = []
        zero_vectors = []
        non_zero_profiles = []
        non_zero_vectors = []
        for profile, vector in zip(profiles, vectors):
            if vector.any():
                non_zero_profiles += [profile]
                non_zero_vectors += [vector]
            else:
                zero_profiles += [profile]
                zero_vectors += [vector]

        log.debug(
            "Profiles: %d, Non-Zero: %d, Zero: %d",
            len(profiles),
            len(non_zero_profiles),
            len(profiles) - len(non_zero_profiles),
        )

        return (non_zero_profiles, non_zero_vectors), (zero_profiles, zero_vectors)


class ProfileTfIdfEmbedder(ProfileEmbedder):
    """
    Creates a vector representation for a profile using Embeddings and Tf-Idf weighting
    """

    def __init__(self, embedder: TokenEmbeddings, use_tf_idf: bool = True):
        self._use_tf_idf = use_tf_idf
        super().__init__(embedder)

    def embed(
        self, profiles: ProfileCollection
    ) -> Tuple[Tuple[List[Profile], List[NDArray]], Tuple[List[Profile], List[NDArray]]]:
        """
        Embeds a Profile collection using tf-idf weighted embeddings
        :param profiles: Profiles to embed
        :return: A tuple containing (profiles with nonzero vectors, vectors),(profiles with zero vectors, vectors)
        """

        product_idf = compute_idf(profiles)
        vectors = [
            self.embed_single(profile, product_idf=product_idf)
            for profile in tqdm(profiles, desc="Embedding Profiles")
        ]

        zero_profiles = []
        zero_vectors = []
        non_zero_profiles = []
        non_zero_vectors = []
        for profile, vector in zip(profiles, vectors):
            if vector.any():
                non_zero_profiles += [profile]
                non_zero_vectors += [vector]
            else:
                zero_profiles += [profile]
                zero_vectors += [vector]

        log.debug(
            "Profiles: %d, Non-Zero: %d, Zero: %d",
            len(profiles),
            len(non_zero_profiles),
            len(profiles) - len(non_zero_profiles),
        )

        return (non_zero_profiles, non_zero_vectors), (zero_profiles, zero_vectors)

    def embed_single(self, profile: Profile, **kwargs) -> NDArray:
        """
        Computes a vector representing a profile
        :param product_idf: inverse document frequencies for all products
        :param profile: Profile to compute the vector for
        :return: Vector representing a profile
        """
        product_idf: Dict[str, float] = kwargs.get("product_idf", defaultdict(lambda: 0))

        products = [component for component in profile.components if isinstance(component, ProductComponent)]
        product_sentences = [Sentence(product.product.text, use_tokenizer=SegtokTokenizer()) for product in products]
        product_term_frequencies = compute_tf(products)

        batch_size = 4
        for current_batch in batch(product_sentences, batch_size):
            self.embedder.embed(current_batch)

        product_embeddings = []
        tf_idf = [
            product_term_frequencies[product.product.text] * product_idf[product.product.text] for product in products
        ]
        for product_sentence in product_sentences:
            product_embeddings += [
                np.mean(
                    np.column_stack([token.embedding.detach().cpu().numpy() for token in product_sentence]), axis=1
                )
            ]

        profile_embedding = np.zeros(self.embedder.embedding_length)
        if len(product_embeddings) > 0:
            profile_embedding = np.array(
                np.average(np.column_stack(product_embeddings), axis=1, weights=tf_idf if self._use_tf_idf else None)
            )

        if profile_embedding.shape != (self.embedder.embedding_length,):
            raise RuntimeError("The created vector does not have the expected length.")

        return profile_embedding


def compute_tf(products: List[ProductComponent]) -> Dict[str, float]:
    """
    Computes the augmented term frequency for products to prevent bias towards large profiles

    :param products: products to compute the term frequencies for
    :return: a dictionary of products and term frequencies
    """
    if len(products) == 0:
        return {}

    tf_dict_one: Dict[str, float] = defaultdict(lambda: 0)
    for prod in products:
        tf_dict_one[prod.product.text] += 1

    max_freq = np.max(list(tf_dict_one.values()))
    return {key: (0.5 + (0.5 * value / max_freq)) for key, value in tf_dict_one.items()}


def compute_idf(profiles: ProfileCollection) -> Dict[str, float]:
    """
    Computes the idf values for Products. Each Profile is seen as document
    :param profiles: Profiles to compute the idf values for
    :return: dictionary mapping products to idf values
    """
    return {key: math.log(len(profiles) / value) for key, value in document_frequencies(profiles).items()}


def document_frequencies(profiles: Iterable[Profile]) -> Dict[str, float]:
    """
    Compute the document frequencies for all products
    :param profiles:
    :return:
    """
    document_freqs: Dict[str, float] = defaultdict(lambda: 0)

    for profile in profiles:
        product_freqs = product_frequencies(profile)
        for key in product_freqs.keys():
            document_freqs[key.strip()] += 1.0

    return document_freqs


def product_frequencies(profile: Profile) -> Dict[str, float]:
    """
    Counts the number of product mentions in a profile
    :param profile: Profile to count products for
    :return: Dictionary containing a product to count mapping
    """
    product_freqs: Dict[str, float] = defaultdict(lambda: 0)

    for component in profile.components:
        if isinstance(component, ProductComponent):
            product_freqs[component.product.text.strip()] += 1.0

    return product_freqs


def batch(iterable, batch_size=1):
    """
    Helper function to batch an iterable into equal chunks
    :param iterable: Iterable to batch
    :param batch_size: batch size
    :return: iterable of length batch_size
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]


def get_embedder(embeddings: str, embeddigns_type: str = ProfileEmbedder.MODEL) -> TokenEmbeddings:
    """
    extracts the TokenEmbeddings from a SequenceTagger model
    :param embeddigns_type: Embeddings type to load Model uses embeddings provided by SequenceTagger
    :param embeddings: Name of embeddings or path
    :return: the TokenEmbeddings of the Model.
    """
    if embeddigns_type == ProfileEmbedder.MODEL:
        embedder: TokenEmbeddings = SequenceTagger.load(embeddings).embeddings
    elif embeddigns_type == ProfileEmbedder.WORD:
        embedder = WordEmbeddings(embeddings)
    elif embeddigns_type == ProfileEmbedder.FLAIR:
        embedder = FlairEmbeddings(embeddings)
    elif embeddigns_type == ProfileEmbedder.TRANSFORMER:
        embedder = TransformerWordEmbeddings(embeddings)
    elif embeddigns_type == ProfileEmbedder.LOWERCASED_FASTTEXT:
        embedder = LowercaseFastTextEmbeddings(embeddings)
    else:
        raise ValueError(f"Embeddings type {embeddigns_type} does not exist.")

    return embedder


def get_profile_embedder(
    model_path: str,
    embeddings: str,
    use_tf_idf: bool = True,
    contextualize: bool = False,
    embeddigns_type: str = ProfileEmbedder.MODEL,
) -> ProfileEmbedder:
    """
    Creates a Profile Embedder from the given parameters
    :param contextualize: If True a ProfileTransformerEmbedder will be created. In this case use_tf_idf is ignored
    :param use_tf_idf: Whether to use tf_idf weighting (Only works with contextualize=False)
    :param model: SequenceTagger model to use for instantiation
    :return: The created ProfileEmbedder
    """

    embedder = get_embedder(embeddings, embeddigns_type)
    if not contextualize:
        log.info("Using ProfileTfIdfEmbedder")
        profile_embedder: ProfileEmbedder = ProfileTfIdfEmbedder(embedder, use_tf_idf=use_tf_idf)
    else:
        model = SequenceTagger.load(model_path)
        log.info("Using ProfileTransformerEmbedder")
        if use_tf_idf:
            log.warning("Using contextualized embeddings. Ignoring option use_tf_idf")

        if not isinstance(model.embeddings, TransformerWordEmbeddings):
            raise ValueError("Contextualized embeddings are only supported for TransformerWordEmbeddings")

        profile_embedder = ProfileTransformerEmbedder(model)

    return profile_embedder
