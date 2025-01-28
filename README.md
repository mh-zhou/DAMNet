class_indices.json:  
This file contains the names and indices of classes derived from the ADNI dataset primarily used in our research.

DAMNet-3D.ipynb:  
This Jupyter notebook documents the main 3D Alzheimer's disease imaging model and its training outcomes.

model-2D.py:  
This is the source code for our proposed model, DAMNet, specifically designed for Alzheimer's disease imaging.

ranksvm.py:  
This script implements the Approximate Rank Pooling method, which is utilized to convert 3D Alzheimer's images into 2D dynamic sequences.

prune-main.py:  
This file contains the code used for model pruning essential to optimizing our network's performance.

---

Given that the training weight from our 2D model are employed in the transfer learning for our 3D image classification, we provide a link to download this weight. The primary training weight for our 2D model can be accessed at https://drive.google.com/file/d/1VJVKy1XfpiUggStLOHPxe9iZRlQ9besu/view?usp=sharing. Thank you for your thorough review and valuable feedback.

To facilitate access for reviewers and interested readers to replicate the results, we have made the raw data from our primary experiments publicly available. The data can be downloaded from Google Drive via the following link(Google Drive): https://drive.google.com/drive/folders/18Wjg3FI9tnxMR__I3B9wBZUmjHLcxB-1?usp=sharing , including both 2D and 3D AD datasets. The data and download links for supplementary experiments are provided in the supplementary materials.

Thanks for the quote.Zhou M, Zheng T, Wu Z, et al. DAMNet: Dynamic mobile architectures for Alzheimer's disease[J]. Computers in Biology and Medicine, 2025, 185: 109517.（中科院TOP，IF=7.8)


@article{ZHOU2025109517,
title = {DAMNet: Dynamic mobile architectures for Alzheimer's disease},
journal = {Computers in Biology and Medicine},
volume = {185},
pages = {109517},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.109517},
url = {https://www.sciencedirect.com/science/article/pii/S0010482524016020},
author = {Meihua Zhou and Tianlong Zheng and Zhihua Wu and Nan Wan and Min Cheng},
keywords = {Alzheimer's disease, DAMNet, 2D and 3D imaging, Parallel intelligence},
abstract = {Alzheimer's disease (AD) presents a significant challenge in healthcare, highlighting the necessity for early and precise diagnostic tools. Our model, DAMNet, processes multi-dimensional AD data effectively, utilizing only 7.4 million parameters to achieve diagnostic accuracies of 98.3 % in validation and 99.9 % in testing phases. Despite a 20 % pruning rate, DAMNet maintains consistent performance with less than 0.2 % loss in accuracy. The model also excels in handling 3D (Three-Dimensional) MRI data, achieving a 95.7 % F1 score within 805 s during a rigorous three-fold validation over 200 epochs. Furthermore, we introduce a novel parallel intelligent framework for early AD detection that improves feature extraction and incorporates advanced data management and control. This framework sets a new benchmark in intelligent, precise medical diagnostics, adeptly managing both 2D (Two-Dimensional) and 3D imaging data.}
}
