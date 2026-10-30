"""
Pure-Python, no-Spark scorer for the sentiment model -- replaces the
embedded SparkSession src/api/app.py used to keep resident for the life of
the process just to answer /predict calls.

Why this exists: the original serving path ran every /predict call through
`model.transform(df)` against that long-lived SparkSession. Fine at low
concurrency, but a real load test (docs/DEPLOYMENT.md) showed it
collapsing well before the target 1,000 concurrent requests. The actual
computation a /predict call needs -- a few hundred hashed-feature lookups
and one dot product -- is trivial; what doesn't scale is that each call
still pays Spark's fixed per-job overhead (logical/physical planning,
Python<->JVM task serialization, driver-side job scheduling) regardless of
how little work the job itself does, and that overhead is paid by the ONE
driver process no matter how large a cluster sits behind it. This module
reimplements exactly what the trained PipelineModel's five stages compute
(Tokenizer -> StopWordsRemover -> HashingTF -> IDF -> LogisticRegression)
in plain Python, so a request never needs a JVM at all.

The exported values this reads (models/active/lightweight_model.json,
written by export_lightweight_model.py) are the ONLY thing that has to
match the real Spark model bit-for-bit -- in particular, the hashing scheme
below has to replicate Spark's HashingTF bucket assignment exactly (see
_hash_term_to_bucket) or predictions would silently drift from what the
Spark model would have produced, with no obvious symptom. That's why
validate_lightweight_model.py exists: it scores the same sample through
BOTH the real Spark model and this module and asserts they agree, instead
of trusting this replication by inspection alone.
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import mmh3


def _java_whitespace_split(text: str) -> List[str]:
    # Spark's plain Tokenizer (not RegexTokenizer) calls Java's
    # String.split("\\s") -- note: "\\s", not "\\s+". Java's split on a
    # single-character pattern does NOT collapse consecutive separators, so
    # "a  b" (two spaces) becomes ["a", "", "b"], not ["a", "b"] -- an empty
    # string is itself a token that goes on to HashingTF. It does drop
    # TRAILING empty strings (Java's default limit=0 behavior), but not
    # leading/middle ones. re.split keeps all of those by default, so
    # stripping only the trailing run below replicates Java's actual
    # behavior rather than Python's more intuitive str.split(), which would
    # silently disagree with Spark on any review with doubled whitespace.
    parts = re.split(r"\s", text)
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _tokenize(text: str) -> List[str]:
    return _java_whitespace_split(text.lower())


# Spark's HashingTF (org.apache.spark.mllib.feature.HashingTF, which
# pyspark.ml.feature.HashingTF wraps) hashes each term with Murmur3_x86_32
# over the term's UTF-8 bytes, seed 42, then reduces mod numFeatures.
# mmh3.hash(..., signed=True) (the default) returns the same signed 32-bit
# int Scala's Int is, and Python's `%` on a positive modulus already
# returns a non-negative result -- unlike Java's `%`, which keeps the
# dividend's sign and is why Spark's own code needs an explicit
# nonNegativeMod() adjustment that Python never has to replicate.
def _hash_term_to_bucket(term: str, num_features: int) -> int:
    return mmh3.hash(term.encode("utf-8"), 42) % num_features


def _hashing_tf(tokens: List[str], num_features: int) -> dict:
    counts: dict = {}
    for token in tokens:
        bucket = _hash_term_to_bucket(token, num_features)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _apply_idf(raw_counts: dict, idf: List[float]) -> dict:
    return {bucket: count * idf[bucket] for bucket, count in raw_counts.items()}


def _predict_from_features(
    features: dict, coefficients: List[float], intercept: float
) -> Tuple[int, float]:
    # Spark's LogisticRegressionModel, for this binary ("binomial") problem,
    # predicts via a linear margin -- intercept plus the dot product of its
    # coefficients with the feature vector -- fed through the logistic
    # (sigmoid) function to get P(class=1). Only buckets present in the
    # sparse `features` dict contribute; everywhere else is an implicit 0,
    # which is why this is a dict-keyed sum rather than a dense dot product
    # over all of `coefficients`.
    margin = intercept + sum(coefficients[bucket] * value for bucket, value in features.items())
    probability_positive = 1.0 / (1.0 + math.exp(-margin))

    if probability_positive > 0.5:
        return 1, probability_positive
    return 0, 1.0 - probability_positive


class LightweightModel:
    """Loads one exported lightweight_model.json and scores review texts."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text())
        self.num_features: int = data["num_features"]
        self.idf: List[float] = data["idf"]
        self.coefficients: List[float] = data["coefficients"]
        self.intercept: float = data["intercept"]

        # Tokenizer always lowercases before this ever runs, so comparing
        # against a pre-lowered stop-word set replicates caseSensitive=False
        # (StopWordsRemover's default, and what train.py's pipeline uses)
        # without re-lowercasing every token on every request. A
        # caseSensitive=True export keeps its original-case stop words,
        # faithfully matching Spark's own (quirky, given tokens arrive
        # already-lowercased) behavior in that configuration.
        case_sensitive: bool = data["case_sensitive"]
        raw_stop_words = data["stop_words"]
        self.stop_words = set(raw_stop_words) if case_sensitive else {w.lower() for w in raw_stop_words}

    def score_batch(self, review_texts: List[str]) -> List[dict]:
        results = []
        for text in review_texts:
            tokens = [t for t in _tokenize(text) if t not in self.stop_words]
            raw_counts = _hashing_tf(tokens, self.num_features)
            features = _apply_idf(raw_counts, self.idf)
            predicted_class, probability = _predict_from_features(
                features, self.coefficients, self.intercept
            )
            sentiment = "positive" if predicted_class == 1 else "negative"
            results.append({"sentiment": sentiment, "probability": round(probability, 4)})
        return results
