# UGSDA-PyTorch

Official PyTorch implementation of the paper **"Domain-Agnostic Crowd Counting via Uncertainty-Guided Style Diversity Augmentation"** accepted at ACM Multimedia 2024.

## Pre-trained Models

We provide the pre-trained models for download via Google Drive:
- [Pre-trained Models on Google Drive](https://drive.google.com/drive/folders/1PYkXg2AcrewGeGLb8jL9io0OSj0OaRkP?usp=sharing)

## Environment Setup

The following environment setup was used to ensure reproducibility:
- Python 3.8
- CUDA Toolkit 11.3.1
- PyTorch 1.11.0
- NumPy 1.23.0
- Matplotlib 3.6.2
- Pandas 2.0.3
- Pillow 9.4.0

Ensure these dependencies are installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Testing Pre-trained Models

We provide code to test our models. The provided models `SHHA_parameter.pth` and `SHHB_parameter.pth` were trained on the ShanghaiTech Part A (SHHA) and Part B (SHHB) datasets, respectively.

To visualize model performance on sample images, run the following command:

```bash
python test_vis_single.py
```

Visualizations for selected results are included in the `images` folder. You can modify the script to test on other images.

### Testing on Public/Custom Dataset 

We also provide a method to test on public/custom datasets. The dataset should be organized in the following structure:

```
└── datasets
    └── dataset_name
        └── test
            ├── den
            │   ├── 1.csv
            │   ├── 2.csv
            │   └── ...
            └── img
                ├── 1.jpg
                ├── 2.jpg
                └── ...
```

Once your dataset is properly structured, you can run the following command to test:

```bash
python test.py
```

For detailed explanations of the network, please refer to our paper.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{ding2024domain,
  title={Domain-Agnostic Crowd Counting via Uncertainty-Guided Style Diversity Augmentation},
  author={Ding, Guanchen and Liu, Lingbo and Chen, Zhenzhong and Chen, Chang Wen},
  booktitle={ACM Multimedia 2024}
}
```

## Acknowledgements

This codebase builds on and acknowledges the following repositories:
- [C-3 Framework](https://github.com/gjy3035/C-3-Framework)

We thank the authors of these repositories for their contributions to the community.

## Contact

If you have any questions or issues, please feel free to reach out to me at: [guanchen.ding@connect.polyu.hk](mailto:guanchen.ding@connect.polyu.hk)

