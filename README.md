# Position-Aware Unified Embedding with Linear Attention for Distinguishable Flow Modeling

[Tao Cui](https://github.com/laibajiu/STLAformer/), Yudong Lu, Di Dong, Chongguang Ren, Zhijian Qu, Panjing Li. Transportation Research Part C: Emerging Technologies, 2026.  
  
This is a PyTorch implementation of **STLAformer**, as described in our paper:  
[https://doi.org/10.1016/j.trc.2025.105423](https://doi.org/10.1016/j.trc.2025.105423)

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

## Required Packages
```bibtex
pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo
```

---

## PeMSD3 & PeMSD4 & PeMSD7 & PeMSD8
- Download the data [百度网盘](https://pan.baidu.com/s/1UC6SWLck1oiiUt7ikQedVw?pwd=ftti) or [Google Drive](https://drive.google.com/file/d/1bcXorzSz3N_58K9A424jL-wADNgv7qE9/view?usp=drive_link).
- Unzip them to corresponding folders.

---

## Training Commands
```bibtex
cd model/  
python train.py --d {dataset} --g {gpu_id}  
```

---

## Testing Commands
```bibtex
Select TestPreTrained.py, choose the dataset to be tested, and import the trained model parameters from {DATASET_NAME}.pth in the pre-trained folder.  
```


