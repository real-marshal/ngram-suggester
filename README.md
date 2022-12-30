# ngram_suggester

**Simple CLI word suggester based on ngram language models**

## Usage

1. Make sure you have `poetry` installed.
2. `poetry install` to create a venv and install the necessary packages.
3. Run the following commands to download datasets (currently one of _brown_, _gutenberg_, _reuters_ is needed):

```
poetry shell
python
import nltk
nltk.download()
```

4. `poetry run train` to open a CLI for training the model
5. `poetry run cli`

Enter a word and a space to show suggestions, continue till the end of a sentence.

Keep in mind that it is case sensitive. Using 'i' instead of 'I' will lead to stupid behavior.

One of the most obvious improvements that could be done here is to also use lower order ngrams for cases where counts are too small (instead of just adding them if there's still space).
