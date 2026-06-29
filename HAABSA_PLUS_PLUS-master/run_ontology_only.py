import numpy as np

from config_ont import FLAGS, YEAR
from OntologyReasoner_main import OntReasoner


def main():
    print(f"\n=== Ontology-only run for SemEval {YEAR} ===")
    print(f"Test data:        {FLAGS.test_path_ont}")
    print(f"Remaining .txt:   {FLAGS.remaining_test_path}")

    ont = OntReasoner()
    accuracy, n_remaining = ont.run(
        use_backup=True,
        path=FLAGS.test_path_ont,
        use_svm=False,
    )

    line_indices = ont.remaining_pos_vector
    instance_indices = np.unique((line_indices // 3).astype(int))

    csv_path = f"rem{YEAR}.csv"
    np.savetxt(csv_path, instance_indices, fmt="%d")

    # --- Sanity stats ---
    # Count total instances in the source file
    with open(FLAGS.test_path_ont) as f:
        total_lines = sum(1 for _ in f)
    total_instances = total_lines // 3
    n_conclusive = total_instances - len(instance_indices)

    print("\n=== Results ===")
    print(f"Total test instances:        {total_instances}")
    print(f"Conclusive (ontology-decided): {n_conclusive}")
    print(f"Inconclusive (backup needed):  {len(instance_indices)}")
    print(f"Ontology accuracy on conclusive: {accuracy:.6f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {FLAGS.remaining_test_path}")


if __name__ == "__main__":
    main()
