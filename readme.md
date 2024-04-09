## CECL

## 0.Citation


If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wan2024unlocking,
  title={Unlocking the Power of Open Set: A New Perspective for Open-Set Noisy Label Learning},
  author={Wan, Wenhai and Wang, Xinrui and Xie, Ming-Kun and Li, Shao-Yuan and Huang, Sheng-Jun and Chen, Songcan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15438--15446},
  year={2024}
}
```
## 1.training

### (1) CIFAR80N
#### Training

- Before training with CECL, initialize training using Promix[https://github.com/Justherozen/ProMix]. Place the obtained correction records and labels into the res_stage1 folder. Then, run:

```
sh run.sh
```

- This is a demo under the cifar80N sym20% setting. You can directly use `sh run.sh` to start training. If you want to customize the training, please go to the `config` folder and modify the relevant parameters.

## 2. Requirements
* To install requirements: 
```
pip install -r requirements.txt
```
* Run in linux (may have some problems in windows)

