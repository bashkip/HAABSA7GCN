from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import stanza
from pytorch_transformers import BertTokenizer
from tqdm import tqdm


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPORA = [
    "train2015restaurant.txt",
    "test2015restaurant.txt",
    "train2016restaurant.txt",
    "test2016restaurant.txt",
]

_BERT_TOK = BertTokenizer.from_pretrained("bert-large-uncased")


def load_instances(filepath: Path) -> Tuple[List[List[str]], int, int]:

    instances: List[List[str]] = []
    n_filtered_tokens = 0
    n_affected_instances = 0
    with filepath.open("r") as fh:
        lines = fh.readlines()
    for i in range(0, len(lines), 3):
        text_left, _, text_right = (
            s.lower().strip() for s in lines[i].partition("$T$")
        )
        aspect = lines[i + 1].lower().strip()
        # Mirror the HAABSA4GCN baseline's substitution for any extra $T$ occurrences in the right
        # side of the sentence.
        text_right = text_right.replace("$T$", aspect)
        sentence = f"{text_left} {aspect} {text_right}".strip()
        raw_tokens = [tok for tok in sentence.split(" ") if tok]
        # Keep only tokens that BERT will produce >=1 subwords for.
        # Mirrors the silent-drop behaviour of ABSADataset.ws() so the leaf
        # count of the resulting constituency tree matches the BERT-side
        # token_head_list. Per-instance drop count is tallied for the receipt
        # line in process_corpus.
        tokens = [tok for tok in raw_tokens if _BERT_TOK.tokenize(tok)]
        dropped = len(raw_tokens) - len(tokens)
        if dropped:
            n_filtered_tokens += dropped
            n_affected_instances += 1
        instances.append(tokens)
    return instances, n_filtered_tokens, n_affected_instances


def build_pipeline() -> stanza.Pipeline:
    
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        tokenize_pretokenized=True,
        use_gpu=False,
        verbose=False,
    )


def parse_instances(nlp: stanza.Pipeline, instances: List[List[str]]) -> Iterable[Tuple[List[str], str]]:

    for tokens in tqdm(instances, desc="parsing", unit="sent"):
        doc = nlp([tokens])
        sent = doc.sentences[0]
        # stanza.models.constituency.parse_tree.Tree.__repr__ returns a
        # single-line PTB-bracketed string.
        tree_str = str(sent.constituency)
        yield tokens, tree_str


def write_const_file(output_path: Path, parses: Iterable[Tuple[List[str], str]]) -> int:
    
    count = 0
    with output_path.open("w") as fh:
        for _, tree_str in parses:
            fh.write(tree_str.rstrip("\n"))
            fh.write("\n\n")
            count += 1
    return count


def process_corpus(
    nlp: stanza.Pipeline,
    input_path: Path,
    output_path: Path | None = None,
    limit: int | None = None,
    to_stdout: bool = False,
) -> None:
    instances, n_filtered, n_affected = load_instances(input_path)
    if limit is not None:
        instances = instances[:limit]
    print(
        f"[{input_path.name}] {len(instances)} instances "
        f"({'sample' if limit else 'full'})",
        file=sys.stderr,
    )

    if n_filtered:
        tok_word = "token" if n_filtered == 1 else "tokens"
        inst_word = "instance" if n_affected == 1 else "instances"
        print(
            f"[const-build] {input_path.stem}: filtered {n_filtered} "
            f"{tok_word} in {n_affected} {inst_word}",
            file=sys.stderr,
        )
    else:
        print(
            f"[const-build] {input_path.stem}: filtered 0 tokens",
            file=sys.stderr,
        )

    parses = parse_instances(nlp, instances)

    if to_stdout:
        for tokens, tree_str in parses:
            print(f"# tokens ({len(tokens)}): {' '.join(tokens)}")
            print(tree_str)
            print()
        return

    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + ".const")
    t0 = time.time()
    n = write_const_file(output_path, parses)
    print(
        f"[{input_path.name}] wrote {n} parses -> {output_path} "
        f"in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="single .txt to parse")
    src.add_argument(
        "--all",
        action="store_true",
        help=f"parse all four corpora in {DEFAULT_DATA_DIR}",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .const path (default: <input>.const)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="parse only the first N instances (for quick sampling)",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="print parses to stdout instead of writing a .const file",
    )
    args = p.parse_args()

    nlp = build_pipeline()

    if args.all:
        for name in CORPORA:
            process_corpus(
                nlp,
                DEFAULT_DATA_DIR / name,
                output_path=None,
                limit=args.limit,
                to_stdout=args.stdout,
            )
    else:
        process_corpus(
            nlp,
            args.input,
            output_path=args.output,
            limit=args.limit,
            to_stdout=args.stdout,
        )


if __name__ == "__main__":
    main()
