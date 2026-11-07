"""Deterministic keyword and bigram frequencies over the exported CSV — no LLM, no vectors.

A cheap sanity check on what the LLM reports claim: if a theme is "top", it should show up here too.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from chat_insight import config

DEFAULT_INPUT = config.EXPORT_CSV
DEFAULT_TOP = 20

# Words that signal frustration/complaint/problem context
FRUSTRATION_SEEDS = {
    "problem",
    "issue",
    "broken",
    "crash",
    "crashed",
    "ridiculous",
    "joke",
    "annoying",
    "frustrat",
    "frustrating",
    "terrible",
    "awful",
    "disaster",
    "nightmare",
    "useless",
    "waste",
    "fail",
    "failed",
    "failure",
    "concern",
    "worried",
    "worry",
    "wrong",
    "error",
    "bug",
    "slow",
    "impossible",
    "unacceptable",
    "rubbish",
    "disgrace",
    "shambles",
    "mess",
    "struggle",
    "struggling",
    "difficult",
    "difficulty",
    "poor",
    "lack",
    "lacking",
    "inadequate",
    "insufficient",
    "barrier",
    "barriers",
    "stuck",
    "overload",
    "overloaded",
    "burnout",
    "exhausted",
    "fed up",
    "enough",
    "absurd",
    "mad",
    "madness",
}

# Generic English stopwords (no heavy NLP library needed)
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "that",
    "this",
    "it",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "not",
    "no",
    "so",
    "if",
    "as",
    "up",
    "out",
    "all",
    "also",
    "just",
    "about",
    "i",
    "we",
    "you",
    "they",
    "he",
    "she",
    "me",
    "us",
    "them",
    "my",
    "our",
    "your",
    "their",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "than",
    "then",
    "more",
    "very",
    "too",
    "there",
    "here",
    "its",
    "it's",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "won't",
    "would've",
    "i'm",
    "we're",
    "they're",
    "he's",
    "she's",
    "that's",
    "i've",
    "we've",
    "let",
    "get",
    "got",
    "going",
    "go",
    "think",
    "know",
    "see",
    "say",
    "said",
    "thought",
    "like",
    "well",
    "good",
    "one",
    "any",
    "some",
    "many",
    "much",
    "most",
    "into",
    "over",
    "because",
    "now",
    "still",
    "even",
    "only",
    "other",
    "after",
    "before",
    "while",
    "though",
    "although",
    "however",
    "yes",
    "ok",
    "okay",
    "sure",
    "great",
    "really",
    "quite",
    "already",
    "actually",
    "back",
    "https",
    "http",
    "www",
    "co",
    "uk",
    "com",
}


def load_csv(path: str, complaints_only: bool) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p, dtype=str).fillna("")
    df["message"] = df["message"].astype(str)

    # Drop media placeholders, very short messages, URLs-only
    df = df[df["message"].str.len() > 15]
    df = df[~df["message"].str.fullmatch(r"https?://\S+", na=False)]

    if complaints_only:
        pattern = "|".join(re.escape(w) for w in FRUSTRATION_SEEDS)
        mask = df["message"].str.contains(pattern, case=False, regex=True, na=False)
        df = df[mask]
        print(f"  Filtered to {len(df)} complaint-context messages \n")

    return df


# Tokens that appear in shared web page / newsletter scrape artifacts
JUNK_TOKENS = {
    "unsuitable",
    "offensive",
    "voted",
    "vote",
    "upyou",
    "downyou",
    "comment",
    "avatar",
    "label",
    "commented",
    "december",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "skip",
    "navigation",
    "newsletter",
    "linkedin",
    "facebook",
    "twitter",
    "youtube",
    "instagram",
}


def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s'-]", " ", text)  # keep letters, apostrophe, hyphen
    tokens = text.split()
    tokens = [t.strip("'-") for t in tokens]  # strip leading/trailing punctuation
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS and t not in JUNK_TOKENS]
    return tokens


def count_unigrams(messages: list[str]) -> Counter:
    counter: Counter = Counter()
    for msg in messages:
        counter.update(tokenise(msg))
    return counter


def count_bigrams(messages: list[str]) -> Counter:
    counter: Counter = Counter()
    for msg in messages:
        tokens = tokenise(msg)
        for a, b in zip(tokens, tokens[1:]):
            counter[(a, b)] += 1
    return counter


def print_unigrams(counter: Counter, top: int) -> None:
    print(f"{'─' * 50}")
    print(f"  TOP {top} SINGLE WORDS")
    print(f"{'─' * 50}")
    for word, count in counter.most_common(top):
        bar = "█" * min(count // 5, 40)
        print(f"  {word:<25} {count:>5}  {bar}")
    print()


def print_bigrams(counter: Counter, top: int) -> None:
    print(f"{'─' * 50}")
    print(f"  TOP {top} BIGRAMS (word pairs)")
    print(f"{'─' * 50}")
    for (a, b), count in counter.most_common(top):
        phrase = f"{a} {b}"
        bar = "█" * min(count // 2, 40)
        print(f"  {phrase:<30} {count:>5}  {bar}")
    print()


def show_examples(df: pd.DataFrame, terms: list[str], n: int = 3) -> None:
    print(f"{'─' * 50}")
    print("  EXAMPLE MESSAGES PER TOP TERM")
    print(f"{'─' * 50}")
    for term in terms:
        pattern = re.escape(term)
        matches = df[df["message"].str.contains(pattern, case=False, na=False)]
        if matches.empty:
            continue
        print(f"\n  [{term}]  ({len(matches)} message(s))")
        for _, row in matches.head(n).iterrows():
            ts = row.get("timestamp", "")[:16]
            msg = row["message"].replace("\n", " ")[:140]
            print(f"    {ts}  {msg}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chat-insight keywords",
        description="Keyword/phrase frequency analysis over the exported chat CSV.",
        epilog=(
            "Examples:\n"
            "  chat-insight keywords\n"
            "  chat-insight keywords --top 30 --complaints-only\n"
            "  chat-insight keywords --bigrams --show-examples\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help="CSV path (default: %(default)s)")
    p.add_argument("--top", "-n", type=int, default=DEFAULT_TOP, help="Number of top terms (default: %(default)s)")
    p.add_argument(
        "--complaints-only",
        action="store_true",
        help="Filter to messages containing complaint/frustration keywords before counting",
    )
    p.add_argument("--bigrams", action="store_true", help="Show bigrams instead of (or in addition to) unigrams")
    p.add_argument("--both", action="store_true", help="Show both unigrams and bigrams")
    p.add_argument("--show-examples", action="store_true", help="Print up to 3 real messages for each top term")
    args = p.parse_args(argv)

    try:
        df = load_csv(args.input, args.complaints_only)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n  Analysing {len(df)} messages from {args.input}\n")

    messages = df["message"].tolist()

    show_unigrams = not args.bigrams or args.both
    show_bigrams = args.bigrams or args.both

    top_terms: list[str] = []

    if show_unigrams:
        uni = count_unigrams(messages)
        print_unigrams(uni, args.top)
        top_terms = [w for w, _ in uni.most_common(args.top)]

    if show_bigrams:
        bi = count_bigrams(messages)
        print_bigrams(bi, args.top)
        if not top_terms:
            top_terms = [f"{a} {b}" for (a, b), _ in bi.most_common(args.top)]

    if args.show_examples and top_terms:
        show_examples(df, top_terms[:10])

    return 0
