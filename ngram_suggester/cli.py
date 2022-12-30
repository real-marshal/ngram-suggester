import re
from typing import Callable, Optional
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from dill import load
from nltk.lm.api import LanguageModel
from nltk.tokenize import RegexpTokenizer

with open("trained_model.pkl", "rb") as f:
    model_dump = load(f)
    lm: LanguageModel = model_dump["lm"]
    tokenizer: Callable[[str], list[str]] = model_dump["tokenizer"]

n = 3
num_of_suggestions = 8


def get_suggestions(
    words: list[str], prev_suggestions: Optional[list[str]] = None
) -> list[str]:
    # Shared mutable default arguments is such a dumb idea, wtf python
    if prev_suggestions == None:
        prev_suggestions = []

    suggestions = [t[0] for t in lm.context_counts(tuple(words)).most_common(10)]

    if len(set(prev_suggestions + suggestions)) < num_of_suggestions and len(words) > 1:
        suggestions += get_suggestions(words[1:], suggestions)

    # Unique while preserving ordering
    return list(
        filter(
            lambda s: (s != "<UNK>" and s != "<s>"), list(dict.fromkeys(suggestions))
        )
    )[:num_of_suggestions]


def is_space_removed(sugggestion: str, prev_word: str):
    if re.match(r"[\.\,\?\!\'\:\;]", sugggestion):
        return True

    if prev_word == "'":
        return True


class NgramCompleter(Completer):
    def get_completions(self, document, complete_event):
        if document.text[-1] != " ":
            return

        num_of_words_to_consider = n - 1
        words = tokenizer(document.text)[-num_of_words_to_consider - 1 :]
        words_without_periods = [w if w != "." else "<s>" for w in words]

        for suggestion in get_suggestions(words_without_periods):
            yield Completion(
                suggestion,
                start_position=-1 if is_space_removed(suggestion, words[-1]) else 0,
            )


def main():
    prompt("> ", completer=NgramCompleter(), multiline=True)


if __name__ == "__main__":
    main()
