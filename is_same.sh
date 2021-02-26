set -e

python3 dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode > orig.txt
python3 dlrm_small.py --mini-batch-size=2 --data-size=6 --debug-mode > small.txt

diff orig.txt small.txt
