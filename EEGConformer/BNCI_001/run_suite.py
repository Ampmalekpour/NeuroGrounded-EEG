# run_suite.py - the WHOLE EEGConformer / BNCI2014-001 suite with one command:
#
#   python run_suite.py                    # tune -> saliency -> csp -> relieff -> mi -> controls -> tables,
#                                          # 2class and 4class, session protocol, seeds 42,43,44
#   python run_suite.py --classes 2class   # any run_all.py option can be added (--force, --keep-going ...)
#
# It simply calls the repository's run_all.py with --dataset bnci --models eegconformer. Finished steps are skipped,
# so after an interruption just start it again. Output and logs: <class>/results_session/,
# <class>/logs_session/, report_session/REPORT.md, and reports/BNCI_001_session/SUMMARY.md.
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
cmd = [sys.executable, os.path.join(ROOT, "run_all.py"), "--dataset", "bnci", "--models", "eegconformer"] + sys.argv[1:]
print("$ " + " ".join(cmd))
sys.exit(subprocess.call(cmd, cwd=ROOT))
