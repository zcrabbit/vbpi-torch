# vbpi-torch
Pytorch Implementation of Variational Bayesian Phylogenetic Inference


Use the command line to run the codes

Examples:

In the unrooted/ folder

```python
python main.py --dataset DS1 --psp --empFreq
python main.py --dataset DS1 --psp --nParticles 20 --gradMethod rws --empFreq
```

In the rooted/ folder
```python
python main.py --dataset DENV4 --burnin 2501 --coalescent_type constant --clock_type strict --init_clock_rate 1e-3 --sample_info --psp --empFreq
python main.py --dataset HCV --burnin 251 --coalescent_type skyride --clock_type fixed_rate --init_clock_rate 7.9e-4 --sample_info --psp
```
