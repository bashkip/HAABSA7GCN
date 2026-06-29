import csv
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWL_PATH = HERE.parent / "data" / "ontology.owl"
CSV_PATH = HERE / "test_ontology_keys.csv"

OWL_NS = "http://www.w3.org/2002/07/owl#"
LEX_NS = "http://www.kimschouten.com/sentiment/restaurant#"


def extract_lex_rows(owl_path: Path) -> list[list[str]]:
    tree = ET.parse(owl_path)
    root = tree.getroot()
    rows = []
    for cls in root.iter(f"{{{OWL_NS}}}Class"):
        lexes = [
            el.text.strip()
            for el in cls.findall(f"{{{LEX_NS}}}lex")
            if el.text and el.text.strip()
        ]
        if lexes:
            rows.append(lexes)
    return rows


def write_csv(rows: list[list[str]], out_path: Path) -> None:
    width = max(len(r) for r in rows)
    header = [f"col{i}" for i in range(width)]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row + [""] * (width - len(row)))


if __name__ == "__main__":
    rows = extract_lex_rows(OWL_PATH)
    write_csv(rows, CSV_PATH)
    print(f"Wrote {len(rows)} classes x {max(len(r) for r in rows)} cols -> {CSV_PATH}")
