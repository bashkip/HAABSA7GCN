import os
import types


YEAR = 2015 # 2016

# ---- repo-relative paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH_ONT = os.path.join(HERE, "Data", f"GloVetestdata{YEAR}.txt")

# Where the remaining-instances .txt gets written. The OntologyReasoner writes
# to FLAGS.remaining_test_path when use_backup=True.
REMAINING_TEST_PATH = os.path.join(
    HERE, f"remainingtestdata{YEAR}.txt"
)

# Unused but referenced — give them harmless values.
REMAINING_SVM_TEST_PATH = os.path.join(HERE, f"remainingsvmtestdata{YEAR}.txt")
TEST_PATH = TEST_PATH_ONT
TEST_SVM_PATH = TEST_PATH_ONT

FLAGS = types.SimpleNamespace(
    year=YEAR,
    test_path_ont=TEST_PATH_ONT,
    test_path=TEST_PATH,
    test_svm_path=TEST_SVM_PATH,
    remaining_test_path=REMAINING_TEST_PATH,
    remaining_svm_test_path=REMAINING_SVM_TEST_PATH,
)
