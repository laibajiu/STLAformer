# [Transportation Research Part C] [Position-Aware Unified Embedding with Linear Attention for Distinguishable Flow Modeling](https://doi.org/10.1016/j.trc.2025.105423)

This is a PyTorch implementation of **STLAformer** for Traffic Forecasting, as described in our paper:  
[Tao Cui](https://github.com/laibajiu/STLAformer/), Yudong Lu, Di Dong, Chongguang Ren, Zhijian Qu, Panjing Li,
Transportation Research Part C: Emerging Technologies, 2026.

---

<p align="center">
  <img src="https://github.com/laibajiu/STLAformer/blob/main/images/Framework%20of%20STLAformer.png?raw=true" alt="Framework of STLAformer" width="100%">
</p>

---

## Performance on Traffic Forecasting Benchmarks
<p align="center">
  <img src="https://github.com/laibajiu/STLAformer/blob/main/images/Evaluations%20of%20STLAformer.png?raw=true" alt="Evaluations of STLAformer" width="100%">
</p>

---

## Citation
```bibtex
@article{cui2026linear,
  title={Position-Aware Unified Embedding with Linear Attention for Distinguishable Flow Modeling},
  author={Cui, Tao and Lu, Yudong and Dong, Di and Ren, Chongguang and Qu, Zhijian and Li, Panjing},
  journal={Transportation Research Part C: Emerging Technologies},
  pages={105423},
  year={2026},
  publisher={Elsevier}
}
```

---

<p>
## Required Packages
```bibtex
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```bibtex
</p>

---

## Training Commands
cd model/
python train.py -d <dataset> -g <gpu_id>

---

<p>
## Testing Commands
Select TestPreTrained.py, choose the dataset to be tested, and import the trained model parameters from {DATASET_NAME}.pth in the pre-trained folder.
</p>

---

## Dataset
PEMS03
PEMS04
PEMS07
PEMS08
