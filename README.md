# NeurIPS 2023 FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning

Zhuo Huang<sup>1</sup>, Li Shen<sup>2</sup>, Jun Yu<sup>3</sup>, Bo Han<sup>4</sup>, Tongliang Liu<sup>1</sup>

<sup>1</sup>The University of Sydney, <sup>2</sup>JD Explore Academy, <sup>3</sup>University of Science and Technology of China, <sup>4</sup>Hong Kong Baptist University

<div align=center>
<img width=600 src=illustration.png/>
</div>

## Abstract
Semi-Supervised Learning (SSL) has been an effective way to leverage abundant unlabeled data with extremely scarce labeled data. However, most SSL methods are commonly based on instance-wise consistency between different data transformations. Therefore, the label guidance on labeled data is hard to be propagated to unlabeled data. Consequently, the learning process on labeled data is much faster than on unlabeled data which is likely to fall into a local minima that does not favor unlabeled data, leading to sub-optimal generalization performance. In this paper, we propose FlatMatch which minimizes a cross-sharpness measure to ensure consistent learning performance between the two datasets. Specifically, we increase the empirical risk on labeled data to obtain a worst-case model which is a failure case that needs to be enhanced. Then, by leveraging the richness of unlabeled data, we penalize the prediction difference (i.e., cross-sharpness) between the worst-case model and the original model so that the learning direction is beneficial to generalization on unlabeled data. Therefore, we can calibrate the learning process without being limited to insufficient label information. As a result, the mismatched learning performance can be mitigated, further enabling the effective exploitation of unlabeled data and improving SSL performance. Through comprehensive validation, we show FlatMatch achieves state-of-the-art results in many SSL settings.

## Running the Experiments
This is an PyTorch implementation of FlatMatch. All our baseline is based on the implementation of [USB framework](https://github.com/microsoft/Semi-supervised-learning). Due to our algorithm requires two-time backward propagation which is incompatible with the APIs of USB, hence we implement our method based on an unofficial FreeMatch repository ([freematch-pytorch](https://github.com/shreejalt/freematch-pytorch)). We would like to thank the authors of both repositories.

### Setup

1. `git clone https://github.com/tmllab/2023_NeurIPS_FlatMatch`
2. `cd 2023_NeurIPS_FlatMatch && install_anaconda.sh`
3. `conda env create -f environment.yml`

### Running the scripts

All the config files for CIFAR10 and CIFAR100 are present in the `config` folder. It follows the `yacs` and logging format inspired from [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). You can visit the given link to learn more about `yacs.config.CfgNode` structure. 

The script `main.py` contains argument parser which can be used to overwrite the config file of the experiment. 

```python
	main.py [-h] [--config-file CONFIG_FILE] [--run-name RUN_NAME]
               [--output-dir OUTPUT_DIR] [--log-dir LOG_DIR] [--tb-dir TB_DIR]
               [--resume-checkpoint RESUME_CHECKPOINT] [--cont-train]
               [--validate-only] [--train-batch-size TRAIN_BATCH_SIZE]
               [--test-batch-size TEST_BATCH_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE
                        Path to the config file of the experiment
  --run-name RUN_NAME   Run name of the experiment
  --output-dir OUTPUT_DIR
                        Directory to save model checkpoints
  --log-dir LOG_DIR     Directory to save the logs
  --tb-dir TB_DIR       Directory to save tensorboard logs
  --resume-checkpoint RESUME_CHECKPOINT
                        Resume path of the checkpoint
  --cont-train          Flag to continue training
  --validate-only       Flag for validation only
  --train-batch-size TRAIN_BATCH_SIZE
                        Training batch size
  --test-batch-size TEST_BATCH_SIZE
                        Testing batch size
  --seed SEED           Seed
```

To execute the training, execute the command 

`python3 main.py --config-file config/cifar10/flatmatch_cifar10_10.yaml`

This will start the training by running the `train()` function in `trainer.py`. 


## Citations
If you find our work insightful, please consider citing our paper, thank you so much!
```
@inproceedings{
	huang2023flatmatch,
	title={FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning},
	author={Zhuo Huang and Li Shen and Jun Yu and Bo Han and Tongliang Liu},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023}
}
```

For any further problem, please feel free to contact [zhuohuang.ai@gmail.com](zhuohuang.ai@gmail.com).

