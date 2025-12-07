# A Federated Learning Approach using Pre-trained BART Model for Text Classification and Generation Tasks
![Made With python](https://img.shields.io/badge/Made%20with-Python-brightgreen)![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)![Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-green.svg)

### Abstract
Abstract—This paper proposes a novel approach, which integrates Federated Learning (FL) with pre-trained BART models
for text classification and generation tasks. In particular, we first design FedAvgBART, a framework that adapts the BART architecture for FL. We then fine-tune the pre-trained BART models using FedAvgBART to support both text classification and generation tasks. Comprehensive experiments performed to evaluate the effectiveness and efficiency of this approach on
benchmark datasets using two BART-variants of different sizes, including BART-large and DistilBART. The significant results demonstrate that federated training can surpass centralized fine-tuning in performance. Specifically, BART-large exhibits exceptional proficiency in classification tasks, while DistilBART excels in text generation, offering superior computational efficiency for resource-limited clients. Furthermore, BART-large maintains greater stability across diverse client scales, whereas non-IID
data disproportionately affects smaller models, underscoring the robustness of larger architectures. Our implementation is available in this Github repository.

### Paper
[Download Paper](paper/[Final]-A-Federated-Learning-Approach-using-Pre-trained-BART-Models-for-Text-Classification-and-Generation-Tasks.pdf)

### Citation
If you would like to cite this paper, please use the following reference:
```
@inproceedings{pham2025federated,
  title={A Federated Learning Approach using Pre-trained BART Models for Text Classification and Generation Tasks},
  author={Pham, Quang-Vinh and Le, Xuan-Tuyen and Le, Quang-Hung},
booktitle={17th International Conference on Frontiers of Information Technology, Applications and Tools},
  year={2025}
}
```
