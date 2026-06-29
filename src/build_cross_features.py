from __future__ import annotations

import argparse
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertTokenizer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
BERT_NAME = "bert-large-uncased"
MAX_SEQ_LEN = 128  # comfortable for sentence + aspect + 3 specials
BATCH_SIZE = 16
INSPECT_K = 3
INSPECT_NEIGHBOURS = 3


# XML parsing


def parse_xml_opinions(xml_path: Path) -> List[Dict]:
    out = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for sent in root.iter("sentence"):
        text_el = sent.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text
        ops_el = sent.find("Opinions")
        if ops_el is None:
            continue
        for op in ops_el.findall("Opinion"):
            tgt = op.attrib.get("target", "")
            if tgt == "NULL":
                continue
            out.append({
                "sentence": text,
                "aspect": tgt,
                "category": op.attrib["category"],
                "polarity": op.attrib["polarity"],
                "from": int(op.attrib["from"]),
                "to": int(op.attrib["to"]),
            })
    return out


# Aspect-subword localization on [CLS] sentence [SEP] aspect [SEP]

def encode_with_aspect_span(
    tokenizer: BertTokenizer,
    sentence: str,
    aspect: str,
    frm: int,
    to: int,
) -> Tuple[List[int], List[int], List[int], int, int]:

    before = sentence[:frm]
    asp = sentence[frm:to]
    after = sentence[to:]

    before_toks = tokenizer.tokenize(before) if before.strip() else []
    asp_toks = tokenizer.tokenize(asp)
    after_toks = tokenizer.tokenize(after) if after.strip() else []
    sent_toks = before_toks + asp_toks + after_toks


    full_sent_toks = tokenizer.tokenize(sentence)

    if full_sent_toks != sent_toks:
        asp_start_in_sent = _find_subsequence(full_sent_toks, asp_toks)

        if asp_start_in_sent is None:
            asp_toks_retry = tokenizer.tokenize(asp.strip())
            asp_start_in_sent = _find_subsequence(full_sent_toks, asp_toks_retry)
            if asp_start_in_sent is None:
                raise RuntimeError(
                    f"could not locate aspect {asp!r} subwords in sentence tokenization for "
                    f"sentence={sentence!r}"
                )

            asp_toks = asp_toks_retry
        sent_toks = full_sent_toks
        before_len = asp_start_in_sent
    else:
        before_len = len(before_toks)

    # Layout: [CLS] sentence_toks [SEP] asp_toks [SEP]
    max_sent_len = MAX_SEQ_LEN - 3 - len(asp_toks)
    if len(sent_toks) > max_sent_len:
        sent_toks = sent_toks[:max_sent_len]
        if before_len + len(asp_toks) > len(sent_toks):
            raise RuntimeError(
                f"truncation removed aspect tokens for sentence={sentence!r} aspect={aspect!r}"
            )

    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    sent_ids = tokenizer.convert_tokens_to_ids(sent_toks)
    asp_ids = tokenizer.convert_tokens_to_ids(asp_toks)

    input_ids = [cls_id] + sent_ids + [sep_id] + asp_ids + [sep_id]
    n_sent_segment = 1 + len(sent_ids) + 1  # [CLS] sent [SEP]
    segment_ids = [0] * n_sent_segment + [1] * (len(asp_ids) + 1)
    attention_mask = [1] * len(input_ids)

    # Aspect span inside the sentence portion (the part that's in segment 0)
    asp_start = 1 + before_len               # +1 for [CLS]
    asp_end = asp_start + len(asp_toks)
    return input_ids, segment_ids, attention_mask, asp_start, asp_end


def _find_subsequence(haystack: List[str], needle: List[str]):
    """Return index of `needle` in `haystack`, or None. O(N*K)."""
    if not needle:
        return None
    n, k = len(haystack), len(needle)
    for i in range(n - k + 1):
        if haystack[i:i + k] == needle:
            return i
    return None


def pool_aspect_features(
    bert: BertModel,
    tokenizer: BertTokenizer,
    opinions: List[Dict],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:

    bert.eval()
    hidden = bert.config.hidden_size
    out = torch.zeros(len(opinions), hidden, dtype=torch.float32)

    encoded = []
    for op in opinions:
        ids, segs, mask, a0, a1 = encode_with_aspect_span(
            tokenizer, op["sentence"], op["aspect"], op["from"], op["to"]
        )
        encoded.append((ids, segs, mask, a0, a1))

    t0 = time.time()
    for batch_start in range(0, len(encoded), batch_size):
        batch = encoded[batch_start:batch_start + batch_size]
        max_len = max(len(b[0]) for b in batch)
        ids_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        seg_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        msk_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, (ids, segs, mask, _a0, _a1) in enumerate(batch):
            L = len(ids)
            ids_t[i, :L] = torch.tensor(ids, dtype=torch.long)
            seg_t[i, :L] = torch.tensor(segs, dtype=torch.long)
            msk_t[i, :L] = torch.tensor(mask, dtype=torch.long)
        ids_t = ids_t.to(device)
        seg_t = seg_t.to(device)
        msk_t = msk_t.to(device)
        with torch.no_grad():
            seq_out, _pooled = bert(ids_t, token_type_ids=seg_t, attention_mask=msk_t)
        # mean-pool aspect-subword positions per item
        for i, (_ids, _segs, _mask, a0, a1) in enumerate(batch):
            out[batch_start + i] = seq_out[i, a0:a1, :].mean(dim=0).cpu().float()
        if (batch_start // batch_size) % 10 == 0:
            done = batch_start + len(batch)
            dt = time.time() - t0
            rate = done / max(dt, 1e-3)
            eta = (len(encoded) - done) / max(rate, 1e-3)
            print(f"    [{done:5d}/{len(encoded):5d}]  {rate:5.1f}/s  eta {eta:5.0f}s", flush=True)
    return out

# Inspection (eyeball sanity)

def inspect_neighbours(year: int, X_train: torch.Tensor, train_ops: List[Dict],
                       categories_train: torch.Tensor, category_vocab: Dict[str, int]) -> None:
    print(f"\n=== qualitative inspection (year={year}) ===")
    Xn = F.normalize(X_train, dim=-1)
    sim = Xn @ Xn.T   # (M, M)
    sim.fill_diagonal_(-float("inf"))
    inv_cat = {v: k for k, v in category_vocab.items()}
    rng = random.Random(7)
    picks = rng.sample(range(len(train_ops)), k=min(INSPECT_K, len(train_ops)))
    for m in picks:
        op = train_ops[m]
        cat_id = categories_train[m].item()
        cat = inv_cat[cat_id]
        print(f"\n  query[{m}] cat={cat}")
        print(f"    sentence : {op['sentence']}")
        print(f"    aspect   : {op['aspect']!r}")
        # top-3 overall
        topk = sim[m].topk(INSPECT_NEIGHBOURS).indices.tolist()
        print(f"    top-{INSPECT_NEIGHBOURS} overall cosine neighbours:")
        for j in topk:
            opj = train_ops[j]
            catj = inv_cat[categories_train[j].item()]
            print(f"      [{j}] cat={catj} aspect={opj['aspect']!r}")
            print(f"           sent: {opj['sentence']}")
        # top-3 same-category
        same_mask = categories_train == cat_id
        same_mask[m] = False
        if same_mask.sum().item() >= INSPECT_NEIGHBOURS:
            sim_masked = sim[m].clone()
            sim_masked[~same_mask] = -float("inf")
            topk_sc = sim_masked.topk(INSPECT_NEIGHBOURS).indices.tolist()
            print(f"    top-{INSPECT_NEIGHBOURS} same-category ({cat}) neighbours:")
            for j in topk_sc:
                opj = train_ops[j]
                print(f"      [{j}] aspect={opj['aspect']!r}")
                print(f"           sent: {opj['sentence']}")


# Main runner

def build_year(year: int, device: torch.device, tokenizer: BertTokenizer,
               bert: BertModel, out_path: Path) -> None:
    print(f"\n========== year={year} ==========")
    train_xml = DATA / f"train{year}restaurant.xml"
    test_xml = DATA / f"test{year}restaurant.xml"
    train_ops = parse_xml_opinions(train_xml)
    test_ops = parse_xml_opinions(test_xml)
    print(f"  train mentions: {len(train_ops)}")
    print(f"  test  mentions: {len(test_ops)}")

    # category vocab from TRAIN only, sorted alphabetically for determinism
    cat_names = sorted({op["category"] for op in train_ops})
    category_vocab = {c: i for i, c in enumerate(cat_names)}
    print(f"  category vocab ({len(category_vocab)}): {cat_names}")

    test_cats_missing = {op["category"] for op in test_ops} - set(category_vocab)
    if test_cats_missing:
        raise RuntimeError(f"test has categories not in train vocab: {test_cats_missing}")

    print(f"  computing x_m for train...")
    X_train = pool_aspect_features(bert, tokenizer, train_ops, device)
    print(f"  computing x_m for test...")
    X_test = pool_aspect_features(bert, tokenizer, test_ops, device)

    cats_train = torch.tensor([category_vocab[op["category"]] for op in train_ops], dtype=torch.long)
    cats_test = torch.tensor([category_vocab[op["category"]] for op in test_ops], dtype=torch.long)

    payload = {
        "train": {
            "X": X_train,
            "categories": cats_train,
            "aspect_terms": [op["aspect"] for op in train_ops],
            "instance_idx": torch.arange(len(train_ops), dtype=torch.long),
        },
        "test": {
            "X": X_test,
            "categories": cats_test,
            "aspect_terms": [op["aspect"] for op in test_ops],
            "instance_idx": torch.arange(len(test_ops), dtype=torch.long),
        },
        "category_vocab": category_vocab,
        "bert_model": BERT_NAME,
        "pooling": "aspect_subword_mean_on_cls_sent_sep_aspect_sep",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"  wrote {out_path}")
    print(f"    X_train: {X_train.shape}  X_test: {X_test.shape}")

    inspect_neighbours(year, X_train, train_ops, cats_train, category_vocab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, choices=[2015, 2016], default=None,
                    help="if omitted, builds both years")
    ap.add_argument("--device", default=None,
                    help="cpu / cuda / mps. default: cuda if available else cpu")
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    print(f"loading {BERT_NAME} tokenizer + model (off-the-shelf, frozen)...")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME, do_lower_case=True)
    bert = BertModel.from_pretrained(BERT_NAME)
    bert.to(device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False
    print(f"  loaded in {time.time() - t0:.1f}s")

    years = [args.year] if args.year else [2015, 2016]
    for year in years:
        out_path = DATA / f"cross_features_{year}.pt"
        build_year(year, device, tokenizer, bert, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
