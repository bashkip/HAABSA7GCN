# HAABSA7GCN

Aspect-based sentiment classification on the SemEval-2015/2016 restaurant
reviews, extending **HAABSA4GCN** (Kazakova et al., 2026) with three additional
graph-convolutional modules and a learned fusion layer.

The four inherited modules — a syntactic dependency GCN (TGCN/SynGCN), a
semantic GCN (SemGCN), a lexical co-occurrence GCN (LexGCN), and an
ontology-knowledge GCN (KnoGCN) — are kept unchanged. HAABSA7GCN adds:

- **ConstGCN** — a constituency-tree GCN (typed edges over Penn Treebank
phrase categories).
- **XCatGCN** — a cross-example *category* GCN: a graph linking each
aspect mention to same-category mentions, over frozen pretrained-BERT node
features.
- **XSimGCN** — a cross-example *similarity* GCN: a top-K cosine-similarity
graph over the same frozen node features.
- **Gated fusion** — a self-conditioned, module-count-agnostic softmax gate
over the per-module aspect representations, replacing plain concatenation.

All new modules are additive and toggleable, so the original 4GCN configuration
runs unchanged.

## Repository layout

```
HAABSA7GCN/
├── README.md
├── cooc_matrix_final2.csv              443 MB lexical co-occurrence matrix (LexGCN input; see note below)
├── data/                               SemEval inputs, cached parses, ontology, hybrid-eval indices
│   ├── {train,test}{2015,2016}restaurant.txt        SemEval sentences/targets/polarities
│   ├── {train,test}{2015,2016}restaurant.txt.dep    cached dependency parses
│   ├── {train,test}{2015,2016}restaurant.txt.const  cached constituency parses (ConstGCN)
│   ├── {train,test}{2015,2016}restaurant.xml        raw SemEval XML (input to build_cross_features.py)
│   ├── cross_features_{2015,2016}.pt                 cached frozen-BERT cross-example node features
│   ├── ontology.owl, ontology.owl-Expanded.owl      domain ontology
│   └── rem{2015,2016}.csv                            ontology-inconclusive instance indices (hybrid eval)
├── src/
│   ├── 7GCN.ipynb                       main notebook: data pipeline, all 7 modules, training, evaluation
│   ├── cross_example.py                 XCatLayer + CrossExampleGraphs (imported by the notebook)
│   ├── fusion.py                        GatedFusion (imported by the notebook)
│   ├── build_const_trees.py            one-shot: writes data/*.txt.const (Stanza constituency parser)
│   ├── build_cross_features.py         one-shot: writes data/cross_features_{year}.pt (frozen BERT)
│   ├── build_ontology_csv.py           one-shot: writes test_ontology_keys.csv from ontology.owl
│   ├── test_ontology_keys.csv          ontology lexical lookup read by the notebook (KnoGCN)
│   ├── dual_model_eval.ipynb           hybrid (ontology + backup) accuracy calculation
│   ├── eval_files/                      per-model backup predictions consumed by dual_model_eval
│   ├── requirements.txt                frozen package list (Python 3.10.14)
│   ├── .python-version                 pyenv pin (3.10.14)
│   ├── figures/                        qualitative-analysis figures
│   ├── (4gcn_vs_7gcn_examples | neighborhood_inspect | per_class_analysis).ipynb   analysis notebooks
│   └── kaggle_runners/                  Kaggle/Colab driver notebooks (see "Kaggle runners")
└── HAABSA_PLUS_PLUS-master/            ontology reasoner subsystem (see "Ontology pipeline")
```

## Environment

Python 3.10.14. The model code depends on the **legacy** `pytorch_transformers`
package (not the modern `transformers`); this is intentional and must not be
migrated, as it preserves the baseline.

```bash
cd src
python3.10 -m venv .venv          # or: uv venv .venv --python 3.10
source .venv/bin/activate
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
```

The ontology subsystem in `HAABSA_PLUS_PLUS-master/` has its own, separate
environment (`requirements.txt` inside that folder) — see below.

## Data and cached artifacts

The expensive preprocessing is cached and shipped with the repo, so training
can run directly:


| Artifact                        | Produced by                  | Consumed by             |
| ------------------------------- | ---------------------------- | ----------------------- |
| `data/*.txt.dep`                | (inherited dependency parse) | dependency GCN          |
| `data/*.txt.const`              | `build_const_trees.py`       | ConstGCN                |
| `data/cross_features_{year}.pt` | `build_cross_features.py`    | XCatGCN / XSimGCN       |
| `src/test_ontology_keys.csv`    | `build_ontology_csv.py`      | KnoGCN                  |
| `data/rem{year}.csv`            | ontology reasoner (below)    | `dual_model_eval.ipynb` |


You only need to re-run a `build_*.py` script if you want to regenerate its
output from scratch (e.g. on a new corpus). Note that `build_cross_features.py`
uses off-the-shelf `bert-large-uncased` and is GPU-intensive.

> **`cooc_matrix_final2.csv` (443 MB).** This lexical co-occurrence matrix is a
> required LexGCN input. It is **not** tracked in this repository — it exceeds
> GitHub's 100 MB per-file limit and is excluded via `.gitignore`. To reproduce
> results you must regenerate it and place it **exactly at the repository root**
> (`./cooc_matrix_final2.csv`, alongside this README), since the notebook loads
> it from that path. It is loaded once and passed into the model as the `cooc`
> argument.

## Running the model

Open `src/7GCN.ipynb`. The notebook is organised top to bottom as: data
pipeline → model building blocks → `AsaTgcnSem` (the hybrid model) → training
loop → run cells. Run the cells in order; the run section near the bottom loads
the co-occurrence matrix and ontology lookup once, then trains.

> **Working directory / paths.** The notebook uses paths relative to its run
> directory (`data/...`, `cooc_matrix_final2.csv`, `test_ontology_keys.csv`)
> and imports the local `cross_example` / `fusion` modules. It was developed to
> run with these files flattened into a single working directory (e.g. a Kaggle
> input). Ensure `data/`, `cooc_matrix_final2.csv`, `test_ontology_keys.csv`,
> `cross_example.py`, and `fusion.py` are all reachable from wherever you launch
> the notebook.

A run is configured through `get_args(...)`, which returns the `opt` config
namespace passed to `main(opt)`. Modules and fusion are selected by arguments:

```python
opt = get_args(
    year='2015',                 # '2015' or '2016'
    seed=7,
    model_type='tri_gcn',
    # --- module toggles (the 4GCN baseline is the first four) ---
    tgcn=True, semgcn=True, lexgcn=True, knogcn=True,
    constgcn=True, xcatgcn=True, xsimgcn=True,
    # --- fusion: 'concat' | 'gate' (legacy 2-module GMU) | 'gated' (HAABSA7GCN) ---
    fusion_type='gated',
    # --- inherited hyperparameters ---
    learning_rate=1.1122448979591838e-05,
    dropout=0.2285714285714286,
    concat_dropout=0.2285714285714286,
    l2reg=0.027059715881067578,
    batch_size=4, num_epoch=15,
    cooc=cooc, onto_words=onto_words,   # loaded once in the run section
)
main(opt)
```

Toggling modules on/off is the only change needed to run an ablation — no code
edits. The full HAABSA7GCN model is `constgcn=xcatgcn=xsimgcn=True` with
`fusion_type='gated'`; the 4GCN baseline is the four inherited modules with
`fusion_type='concat'`.

Set `year` to `'2016'` to run the other dataset.

## Kaggle runners

`src/kaggle_runners/` contains the driver notebooks that produced the thesis
results. Each one reads the main notebook via `nbformat`, executes its
definition cells (`get_args`, `main`, the model classes), and then runs its own
experiment loop on top.

**These notebooks were run on Kaggle with an NVIDIA Tesla T4 GPU.** They are not
runnable locally: they reference Kaggle input paths
(`INPUT_BASE = /kaggle/input/...`), copy the data into `/kaggle/working`, and
`os.chdir` there before running.

**Setup:** upload this repository to Kaggle as a **Dataset**, then attach it to a
GPU (T4) notebook so the runners can pick up `src/7GCN.ipynb`, `data/`,
`cooc_matrix_final2.csv`, `test_ontology_keys.csv`, `cross_example.py`, and
`fusion.py`. Set each runner's `INPUT_BASE` to match your dataset's path before
running.

**Workflow — run in this order:**

1. **TPE hyperparameter search** — `tpe_search_4GCN.ipynb`, then
  `tpe_search_7GCN.ipynb`. A Tree-structured Parzen Estimator (hyperopt) search
   over learning rate, dropout, L2, and (for 7GCN) fusion dropout. Produces a
   ranking of candidate configurations.
  > **Note:** `tpe_search_7GCN.ipynb` was run on **Google Colab** instead of
  > Kaggle, because Kaggle's 30-hour weekly GPU quota was not sufficient for the
  > 7GCN search. It mounts Google Drive and unzips the repo from there rather
  > than reading from a Kaggle dataset; otherwise the workflow is identical.
2. **Final runs** — `final_run_4GCN.ipynb`, then `final_run_7GCN.ipynb`. Take
  the **top-3 configurations** from the search and train them at full epochs
   across both years, selecting the
   validation-best per the reproduction protocol.
3. **Ablations (last)** — `ablations_7GCN.ipynb`. Toggles the new modules
  (ConstGCN / XCatGCN / XSimGCN) and fusion type on and off to isolate each
   module's contribution.

## Evaluation

`src/7GCN.ipynb` reports the backup-model (neural) test accuracy directly.

`src/dual_model_eval.ipynb` computes the **hybrid** accuracy used in the thesis:
it combines the neural backup predictions (stored under `src/eval_files/`) with
the ontology-conclusive predictions, using `data/rem{year}.csv` to identify the
instances the ontology could not resolve (and which therefore fall through to
the backup model).

## Reproduction protocol

For each year, train on seed 7 and report the test prediction from the epoch
with the **highest validation accuracy**.

Inherited 4GCN hyperparameters: `learning_rate ≈ 1.11e-5`, `batch_size = 4`,
`l2reg ≈ 0.0271`, `dropout ≈ 0.229`, `num_epoch = 15`.

## Ontology pipeline

The `data/rem{year}.csv` files (the ontology-inconclusive instance indices used
by the hybrid evaluation) are produced by the ontology reasoner in
`HAABSA_PLUS_PLUS-master/`.

That folder is the **HAABSA++ codebase of Wallaart & Truşcă (2019)**
([https://github.com/ofwallaart/HAABSA](https://github.com/ofwallaart/HAABSA)), included here largely unchanged. The
additions layered on top of it are:

- `OntologyReasoner_main.py` — **the reasoner actually used here.** It is a
patched copy of the original `OntologyReasoner.py`; run this one, not the
original.
- `run_ontology_only.py` — drives `OntologyReasoner_main.py` on the test set
for the year set in `config_ont.py`, and writes both
`remainingtestdata{year}.txt` (the inconclusive instances) and the
`rem{year}.csv` index file the hybrid eval expects.
- `config_ont.py` — paths and year selection for the runner. It expects the OWL
file at `HAABSA_PLUS_PLUS-master/Data/externalData/ontology.owl`.

To regenerate the `rem` files:

```bash
cd HAABSA_PLUS_PLUS-master
python3 -m venv .venv-ont && source .venv-ont/bin/activate
pip install -r requirements.txt
# set YEAR in config_ont.py, then:
python run_ontology_only.py
```

The `rem` files are already cached in `data/`, so this step is only needed to
regenerate them.

## Credits

- **HAABSA4GCN** — Kazakova et al. (2026): the four-module GCN baseline this
work extends.
- **HAABSA++** — Wallaart & Truşcă (2019): the ontology reasoner in
`HAABSA_PLUS_PLUS-master/`.

## Hardware notes

- CPU-only machines (e.g. Apple Silicon) can run the pipeline on small subsets
to verify correctness, but full training is slow.
- Full training was run on GPU (Kaggle T4). The notebook supports
per-epoch checkpointing to fit within time-limited sessions.

