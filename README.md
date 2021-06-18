# BT4MolGen

Back Translation for Molecule Generation.

This project is based on [fairseq](https://github.com/pytorch/fairseq).  
To get help from the official code, see the [original README](./README-original.md).

## Dataset Preparation

1. Molecular Property Improvement Dataset
2. USPTO: Retrosynthesis Prediction Dataset

## Environment Preparation

1. Docker image: `fyabc/bt4molgen:main`

## Data Preprocessing

1. Data source (canonical, without extra '@' chars): [HERE](https://github.com/wengong-jin/hgraph2graph/tree/master/data).
   1. Placed in <./canonical-from-HierG2G>.
2. Data source (non-canonical, with extra '@' chars): [HERE](https://github.com/wengong-jin/iclr19-graph2graph/tree/master/data).
   1. Placed in <./full-from-JTNN>.
3. For our Transformer translation jobs, we need paired validation and test subsets, so we split `train_pairs.txt` into new valid and test files.
   1. Sizes: TODO
4. Tokenization

    ```bash
    cd /d/GitProjects/off-hg2g/fy-exp/data
    python /d/GitProjects/fairseq-v0.8.0/run/molecule/preprocessing/tokenize_smiles.py -t re drd2/train.m1 drd2/train.re.m1 --std
    ```

## Baseline Training

1. HierG2G baseline training

   ```bash
   # Submit a debug job of 'stub-wu2-t-yafan-philly2-torch15-py36-cu101' (docker = 'pytorch1.5-py36-cuda10.1')

   conda create -n mol-hg2g python=3.6
   conda activate mol-hg2g
   conda install -y pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
   conda install rdkit -c rdkit
   python -m pip install networkx

   cd /blob/v-yaf/off-hg2g/
   python setup.py develop

   for d in drd2 qed logp04 logp06; do
      mkdir -p train_processed/${d}/
      python preprocess.py --train data/${d}/train_pairs.txt --vocab data/${d}/vocab.txt --ncpu 16 < data/${d}/train_pairs.txt
      mv tensor* train_processed/${d}/
   done

   # Same for drd2, logp04, logp06
   mkdir -p models/qed/
   nohup python gnn_train.py --train train_processed/qed/ --vocab data/qed/vocab.txt --save_dir models/qed/ > models/qed/train-log.txt 2>&1 &

   for d in drd2 qed logp04 logp06; do
      mkdir -p models/$d/translations
      python decode.py --test data/$d/test.txt --vocab data/$d/vocab.txt --model models/$d/model.5 --num_decode 20 > models/$d/translations/translation-$d-pt5-20.txt
   done
   ```

2. HierG2G BT Training

   ```bash
   cd /blob/v-yaf/off-hg2g/
   mkdir -p fy-exp/data/reversed-baseline/tmp/

   # Prepare reversed data
   for d in drd2 qed logp04 logp06; do
      cut -d ' ' -f 1 data/$d/train_pairs.txt > fy-exp/data/reversed-baseline/tmp/$d-train_pairs-x.txt
      cut -d ' ' -f 2 data/$d/train_pairs.txt > fy-exp/data/reversed-baseline/tmp/$d-train_pairs-y.txt
      paste -d ' ' fy-exp/data/reversed-baseline/tmp/$d-train_pairs-y.txt fy-exp/data/reversed-baseline/tmp/$d-train_pairs-x.txt > fy-exp/data/reversed-baseline/$d-train_pairs.txt

      mkdir -p train_processed/$d-reversed/
      python preprocess.py --train fy-exp/data/reversed-baseline/$d-train_pairs.txt --vocab data/${d}/vocab.txt --ncpu 16 < fy-exp/data/reversed-baseline/$d-train_pairs.txt
      mv tensor* train_processed/$d-reversed/
   done

   # Train reversed HierG2G models, same for drd2, logp04, logp06
   d=qed
   mkdir -p models/$d-reversed/
   nohup python gnn_train.py --train train_processed/$d-reversed/ --vocab data/$d/vocab.txt --save_dir models/$d-reversed/ > models/$d-reversed/train-log.txt 2>&1 &

   # Generate BT data using reversed HierG2G models
   ```

## Evaluation

## BT Training

1. Translation output folder: [HERE](\\msralab\ProjectData\LA\t-yafan\Molecule\log\std-translations\back-translation-data).
2. Train forward and backward baseline models
3. Generate BT data (`x'`) using backward models
4. Make BT dataset: See `make-bt-dataset.py`.
