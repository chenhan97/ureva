# ureva
We modify the code of previous works (https://github.com/ttthy/ure) to implement our experiments.

## Datasets
NYT: We cannoy provide NYT dataset due to copyright, please contact [Diego Marcheggiani]
SemEval: https://www.aclweb.org/anthology/S10-1006/
Input format: same as [sample](https://github.com/diegma/relation-autoencoder/blob/master/data-sample.txt)

There are some vocabulary files needed to generate in advance.
You can use the scripts

bash ure/preprocessing/run.sh
python data/nyt/process.py

## Usage

### Training

UREVA
python -u -m ure.ureva.main --config models/ureva.yml

### Evaluation
```
python -u -m ure.ureva.main   --config models/ureva.yml --mode test
