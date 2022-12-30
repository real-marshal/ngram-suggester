from enum import Enum
from typing import Callable, Optional, Type
from dill import dump
from nltk.corpus import (
    brown,
    gutenberg,
    reuters,
    PlaintextCorpusReader,
    CategorizedCorpusReader,
)
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
)
from nltk.lm import MLE, KneserNeyInterpolated, Vocabulary, StupidBackoff
from nltk.lm.api import LanguageModel
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
import questionary


class DatasetName(Enum):
    BROWN = "brown"
    GUTENBERG = "gutenberg"
    REUTERS = "reuters"


datasets_map = {
    DatasetName.BROWN: brown,
    DatasetName.GUTENBERG: gutenberg,
    DatasetName.REUTERS: reuters,
}

tokenizers_map: dict[DatasetName, Callable[[str], list[str]]] = {
    DatasetName.BROWN: (lambda words: RegexpTokenizer(r"[\w\']+|\.").tokenize(words)),
    DatasetName.GUTENBERG: wordpunct_tokenize,
    DatasetName.REUTERS: wordpunct_tokenize,
}


def train(
    n: Optional[int] = 3,
    num_of_sents: Optional[int] = None,
    unk_cutoff: int = 5,
    # we don't need smoothing for suggestions
    # lm_class: Type[LanguageModel] = StupidBackoff,
    dataset: PlaintextCorpusReader | CategorizedCorpusReader = brown,
):
    sents = dataset.sents()[:num_of_sents]

    train_data, vocab = padded_everygram_pipeline(n, sents)

    lm = MLE(order=n, vocabulary=Vocabulary(vocab, unk_cutoff=unk_cutoff))
    lm.fit(train_data)

    return lm


def save_lm_to_disk(lm: LanguageModel, dataset: DatasetName):
    with open("trained_model.pkl", "wb") as f:
        dump(dict(lm=lm, tokenizer=tokenizers_map[dataset]), f)


def get_input_args():
    # smoothing_algos_map = dict[str, Type[LanguageModel]](
    #     no_smoothing=MLE,
    #     stupid_backoff=StupidBackoff,
    #     kneser_ney=KneserNeyInterpolated,
    # )

    return {
        "n": int(questionary.text("n:", default="3").ask()),
        "num_of_sents": int(
            questionary.text(
                "num of sents to train on (empty for the full dataset):"
            ).ask()
            or 0
        )
        or None,
        "unk_cutoff": int(
            questionary.text(
                "count threshold to consider a word unknown:",
                default="5",
            ).ask()
            or 0
        )
        or 1,
        # lm_class=smoothing_algos_map[
        #     questionary.select(
        #         "smoothing algo:",
        #         choices=["stupid_backoff", "kneser_ney", "no_smoothing"],
        #     ).ask()
        # ],
        "dataset": DatasetName[
            questionary.select(
                "dataset:",
                choices=list(map(lambda e: e.value, DatasetName)),
            )
            .ask()
            .upper()
        ],
    }


def cli():
    train_args = get_input_args()
    dataset = train_args["dataset"]
    train_args["dataset"] = datasets_map[train_args["dataset"]]

    lm = train(**train_args)

    print(lm.counts)

    save_lm_to_disk(lm, dataset)


if __name__ == "__main__":
    # btw turned out python doesn't have block level scoping...
    cli()
