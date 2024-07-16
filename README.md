# Correspondence Pretext Tasks for Goal-oriented Visual Navigation

[Paper]()

## Installation

1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim#installation):
```bash
conda create -n debit python=3.8 cmake=3.14.0
conda activate debit
conda install habitat-sim=0.2.3 headless -c aihabitat -c conda-forge
```
2. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab#installation) with baselines:
```bash
mkdir deps
git clone https://github.com/facebookresearch/habitat-lab -b v0.2.3 deps/habitat-lab
cd deps/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```
3. Clone [CroCo](https://github.com/naver/croco) repo and make it an installable package:
```bash
cd -
git clone https://github.com/naver/croco src/croco
find src/croco -type d -exec touch {}/__init__.py \;
find src/croco/models -name "*.py" -exec sed -ie 's/^from models/from /' {} \;
```
4. Install DEBiT and CroCo:
```bash
pip install -e .
```
5. Download pre-trained weights:
```bash
mkdir -p out/ckpt/hab_bl/imgnav
cd out/ckpt/hab_bl/imgnav
```
| Architecture |                             CroCo + RPEV + PPO(imgnav)                            |
| ------------ | --------------------------------------------------------------------------------- |
|   DEBiT-L    | `curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_large.pth` |
|   DEBiT-B    | `curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_base.pth`  |
|   DEBiT-S    | `curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_small.pth` |
|   DEBiT-T    | `curl -LO https://download.europe.naverlabs.com/navigation/debit/debit_tiny.pth`  |


## Evaluation
```bash
cd -
python scripts/train_eval_ppo.py \
    --run-type eval \
    --exp-config configs/imgnav-gibson-debit.yaml \
    debit=debit_base \
    habitat_baselines.eval_ckpt_path_dir=out/ckpt/hab_bl/imgnav/debit_base.pth
```
