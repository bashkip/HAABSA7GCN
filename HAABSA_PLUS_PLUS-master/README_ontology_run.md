# Ontology-only run — setup & execution

## What this does
Runs the HAABSA++ ontology classifier in isolation (no LCR-Rot-hop++,
no TensorFlow), and writes:
  - `remainingtestdata{YEAR}.txt` — the inconclusive instances
  - `rem{YEAR}.csv` — instance indices for the 4GCN eval notebook

## File layout
Drop the three files into the root of `HAABSA_PLUS_PLUS-master/`:
  HAABSA_PLUS_PLUS-master/
    ├── config_ont.py             ← NEW
    ├── OntologyReasoner_main.py ← NEW
    ├── run_ontology_only.py      ← NEW
    ├── Data/
    │   ├── GloVetestdata2015.txt  (you have these)
    │   └── GloVetestdata2016.txt
    └── data/
        └── externalData/
            └── ontology.owl       ← MUST be here

## Setup (one time)

### 1. Make sure the OWL file is where owlready2 expects it
The patched reasoner looks at `data/externalData/ontology.owl`. The repo I saw
has `data/ontology.owl`. Run from the repo root:

  mkdir -p data/externalData
  cp data/ontology.owl data/externalData/ontology.owl

(If your OWL file is somewhere else, adjust the `onto_path.append(...)` line
near the top of `OntologyReasoner_main.py`.)

### 2. Python environment
You need Python 3.8 or newer. Create a clean venv to avoid polluting your
system Python:

  cd HAABSA_PLUS_PLUS-master
  python3 -m venv .venv-ont
  source .venv-ont/bin/activate
  pip install --upgrade pip
  pip install owlready2 nltk numpy

### 3. NLTK data
The reasoner uses NLTK for POS-tagging and lemmatization. Download once:

  python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

(The last two are needed on newer NLTK versions; harmless if not.)

## Run

### Year 2016
With YEAR=2016 set in config_ont.py (the default):

  python run_ontology_only.py

Expected output (numbers will be approximate):
  Accuracy:  0.86...
  Total test instances:        650
  Conclusive (ontology-decided): ~470
  Inconclusive (backup needed):  ~180
  Wrote rem2016.csv
  Wrote remainingtestdata2016.txt

### Year 2015
Edit `config_ont.py` and change YEAR = 2015, then run again:

  python run_ontology_only.py

Expected:
  Total test instances:        597
  Conclusive (ontology-decided): ~370
  Inconclusive (backup needed):  ~225

Numbers should land near the paper's HAABSA++ ontology-only accuracies of
65.8% (2015) and 78.3% (2016) reported in the 4GCN eval notebook.

## Plug into 4GCN eval

The notebook expects `data/rem2015.csv` / `data/rem2016.csv` (relative to the
notebook). Move them into place:

  mkdir -p ../code/data
  mv rem2015.csv rem2016.csv ../code/data/

(Adjust the path to wherever your 4GCN repo's `data/` directory is.)

## Troubleshooting

- "ontology.owl not found" → step 1 above. owlready2 uses `onto_path` not a
  direct path, so the file must be in one of those directories.

- NLTK LookupError → re-run the download in step 3. On newer NLTK the
  resource names changed (`punkt_tab`, `averaged_perceptron_tagger_eng`).

- Accuracy looks off vs the paper → the window-only negation check may catch
  a handful of cases differently. The numbers should still land within ~0.5pp
  of HAABSA++'s reported ontology-only accuracy. If you need an exact match,
  re-enable the Stanford parser branch (lines deleted from `is_negated`).

- Slow run (>10 min) → the inner loop loads the lemmatizer once but tokenizes
  each word freshly. Expected runtime on M2: 2–8 min per dataset.
