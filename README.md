# vbpi-torch
Pytorch Implementation of Variational Bayesian Phylogenetic Inference

## Dependencies

* [Biopython](http://biopython.org)
* [bitarray](https://pypi.org/project/bitarray/)
* [dendropy](https://dendropy.org)
* [ete3](http://etetoolkit.org)
* [PyTorch](https://pytorch.org/)

You can build and enter a conda environment with all of the dependencies built in using the supplied `environment.yml` file via:

```
conda env create -f environment.yml
conda activate vbpi-torch
```


## Preparation

Unzip `DENV4_constant_golden_run.trees.zip` in the `rooted/data/DENV4` directory.


## Running

These steps will reproduce the experiments in the preprint [A Variational Approach to Bayesian Phylogenetic Inference](http://arxiv.org/abs/2204.07747).

* The evidence lower bounds will be saved to *_test_lb.npy file.
* The KL divergences will be saved to *_kl_div.npy file (if --empFreq is turned on).
* The trained model will be saved to *.pt file.


In the `unrooted/` folder

```bash
python main.py --dataset DS1 --psp --empFreq
python main.py --dataset DS1 --psp --nParticle 20 --gradMethod rws --empFreq
python main.py --dataset flu100 --psp
python main.py --dataset flu100 --psp --supportType mcmc -cf 100000 --maxIter 400000
```

One can also load the checkpoints for testing (e.g., KL computation)
```bash
python main.py --dataset flu100 --psp --supportType mcmc --empFreq --test --datetime "20xx-xx-xx xx:xx:xx.xxxxxx"
```
where the value for --datetime is the datetime for the saved model that you want to test.

See more concrete examples here: [ds1.ipynb](https://github.com/zcrabbit/vbpi-torch/blob/main/unrooted/notebooks/ds1.ipynb), [flu100.ipynb](https://github.com/zcrabbit/vbpi-torch/blob/main/unrooted/notebooks/flu100.ipynb).

In the `rooted/` folder
```bash
python main.py --dataset DENV4 --burnin 2501 --coalescent_type constant --clock_type strict --init_clock_rate 1e-3 --sample_info --psp --empFreq
python main.py --dataset HCV --burnin 251 --coalescent_type skyride --clock_type fixed_rate --init_clock_rate 7.9e-4 --psp
```

See more concrete examples here: [denv4.ipynb](https://github.com/zcrabbit/vbpi-torch/blob/main/rooted/notebooks/denv4.ipynb), [hcv.ipynb](https://github.com/zcrabbit/vbpi-torch/blob/main/rooted/notebooks/hcv.ipynb).
