# QUICKSTART (Phase 2)
## Install
pip install torch numpy pandas pyyaml biopython tqdm scipy

# optional (via pip) or use conda's bioconda channel
# pip install viennarna

## Run
rsync -av phase2_scaffold_ok/ /root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2/
cd /root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2
bash tools/run_m1.sh
bash tools/run_m2_cvae.sh   # or: bash tools/run_m2_cgan.sh
bash tools/run_m3_rl.sh
