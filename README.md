# Correspondence Pretext Tasks for Goal-oriented Visual Navigation

[Paper]()

## Installation

1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim#installation)
```bash
conda create -n debit python=3.9 cmake=3.14.0
conda activate debit
conda install habitat-sim=0.2.3 headless -c conda-forge -c aihabitat
```
2. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab#installation) with baselines
```bash
git clone https://github.com/facebookresearch/habitat-lab -b v0.2.3
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```
3. Clone [CroCo](https://github.com/naver/croco) repo and make it an installable package:
```bash
git clone https://github.com/naver/croco src/croco
find src/croco -type d -exec touch {}/__init__.py \;
```
See [additional installation instructions](https://github.com/naver/croco#preparation) if you want to pre-train CroCo yourself
4. Install DEBiT and CroCo:
```bash
pip install -e .
```
5. *(Optional)* Download [RPEV data]()
6. *(Optional)* Download pre-trained weights:
| Architecture |  CroCo + RPEV + PPO(imgnav) |
| ------------ |  -------------------------- |
|   DEBiT-L    |                             |
|   DEBiT-B    |                             |
|   DEBiT-S    |                             |
|   DEBiT-T    |                             |

## Training

1. CroCo pretraining: See [instructions](https://github.com/naver/croco#pre-training) on the original CroCo repository
2. RPEV pretraining:
```
conda activate debit
python scripts/pretrain_rpev.py
```
3. PPO:
```
python scripts/train_eval_ppo.py
```

## Evaluation
```
python scripts/train_eval_ppo.py --run-type eval
```
