# Model Plan

This explains **why** the sentiment-prediction pipeline is built the way it is, not just what the code does. It's written for someone new to machine-learning pipelines.

## The problem

Each Amazon review comes with a `polarity` label — `1` for negative, `2` for positive — plus free-text `title` and `text`. The goal is to predict that label from the review text: a binary classification problem (is this review negative or positive?).

## Why the title and body are combined

`preprocess.py` joins `title` and `text` into one `review_text` column before anything else happens (`concat_ws(" ", title, text)`, which just uses the title on its own if the body is missing). The reasoning: a review's title is often a strong sentiment signal by itself. Someone who titles a review "Terrible, don't buy!!" has already told you most of what you need to know. Throwing that away would waste a cheap, useful piece of information. Combining it with the body — rather than treating it as a separate input — also keeps every later step of the pipeline unchanged: the model just sees more text per review. `train.py`, `predict.py`, and the API's `/predict` endpoint all work on `review_text`, never on `text` by itself, so the model is always trained and used on the same kind of input.

## The pipeline, stage by stage

Each stage links to the matching cell in [`02_pipeline_walkthrough.ipynb`](../src/notebooks/02_pipeline_walkthrough.ipynb). On GitHub, the link jumps straight to that stage's code and lets you run it; opened locally the link just opens the notebook, which is short and follows the same order as this document.

### 1. Tokenizer

▶ [Run this stage](../src/notebooks/02_pipeline_walkthrough.ipynb#stage-1-tokenizer)

Splits `review_text` into individual words: "This product is great" becomes `["this", "product", "is", "great"]`. Models work with numbers, not sentences, so the first step in any text pipeline is breaking the text into pieces that can eventually become numbers.

### 2. StopWordsRemover

▶ [Run this stage](../src/notebooks/02_pipeline_walkthrough.ipynb#stage-2-stopwordsremover)

Drops extremely common words like "the", "is", and "a" that carry almost no sentiment on their own. Removing them shrinks the vocabulary and lets the model focus on words that actually separate positive reviews from negative ones, like "great", "terrible", or "disappointed".

### 3. HashingTF — turning words into numbers

▶ [Run this stage](../src/notebooks/02_pipeline_walkthrough.ipynb#stage-3-hashingtf)

Converts the remaining words into a fixed-size numeric vector by hashing each word to a bucket number. The alternative — building an explicit list mapping every unique word in the dataset to its own column (`CountVectorizer`) — needs a full pass over the data first and keeps growing as new words show up. Hashing accepts a small, usually harmless risk that two different words land in the same bucket ("collide") in exchange for a fixed memory footprint and no separate vocabulary-building step — a reasonable trade at this dataset's size, around 3.6 million reviews.

### 4. IDF — weighing rare words more heavily

▶ [Run this stage](../src/notebooks/02_pipeline_walkthrough.ipynb#stage-4-idf-inverse-document-frequency)

IDF stands for Inverse Document Frequency. It re-weights the word counts from the previous step so that words appearing in almost every review — even after removing stopwords — count for less, while rarer, more distinctive words count for more. Without this step, "good" and "phenomenal" would be treated as equally important just because they show up the same number of times, even though the rarer word usually tells you more.

### 5. LogisticRegression

▶ [Run this stage](../src/notebooks/02_pipeline_walkthrough.ipynb#stage-5-logisticregression)

The classifier itself. Logistic regression was chosen over something more complex, like a deep neural network, for three reasons: it trains fast even on millions of rows, it's easy to inspect (each feature gets a weight, so you can see which words push a prediction toward positive or negative), and it's a well-understood, reliable starting point for text classification. If a more advanced model is ever needed, only this last stage has to change — the rest of the pipeline stays the same.

## Why hyperparameter tuning, and why Optuna

A single train/test split only tells you how one specific model setup performed on one specific slice of data. It doesn't tell you whether that setup was actually the best choice, or whether the result was partly luck in how the data happened to split. Guessing values for settings like `HashingTF`'s bucket count or `LogisticRegression`'s regularization strength isn't reliable either — some combinations generalize much better than others, and there's no way to know which without actually trying them.

`train.py` searches for good values using **Optuna**, a hyperparameter optimization library. The earlier approach (still visible in `docs/MODEL_HISTORY.md`'s v1–v3 entries) tried a small, fixed grid of hand-picked combinations — 8 in total — and checked each one with k-fold cross-validation. Optuna instead works adaptively: each *trial* tries one set of values, and every later trial's choice is informed by what every earlier trial scored, using a search strategy called TPE (Tree-structured Parzen Estimator). This lets it search a much wider, continuous range of values — not just a handful of fixed points — while still converging on good settings, typically needing fewer wasted trials than trying every combination in a large grid would.

To keep each trial affordable, the default is to score a trial with a single train/validation split rather than k-fold cross-validation — one fit and one check, instead of several. That's a noisier estimate per trial than k-fold gives, but the tradeoff is deliberate: it means many more trials fit in the same amount of time, and Optuna's adaptive search makes good use of that extra volume. `train.py --cv-folds N` (default 1, meaning "off") switches every trial to real k-fold cross-validation instead — worth turning on mainly when training on a small sample, where a single validation split would be too small to trust, at the cost of each trial taking roughly N times longer.

## What "validation" means during tuning

- The 80% **training** portion is what every trial (and the final winner) actually fits on.
- The 20% **validation** portion is what every trial is scored against. Because Optuna uses that score to decide what to try next, this split is now part of the search itself, not a one-time check performed only at the end.
- The separate `test.parquet` file, used only by `evaluate.py`, is still never touched during training or tuning. It exists purely to report one final, unbiased score once a model is finished — a number from data the model's development never influenced in any way.

The held-out test set is what keeps the final reported number honest — everything before that point (training and validation both) is fair game for the search process to use.

## Where MLflow fits in

Each Optuna search here fits a few dozen separate models (30 by default, one per trial). Comparing that many sets of results by hand doesn't scale, so MLflow logs every trial's settings and score automatically. That log is also what `evaluate.py` uses to build its "did tuning actually help" chart, comparing baseline and best-tuned accuracy and AUC side by side.
