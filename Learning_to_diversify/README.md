# Learning_to_diversify
This is the official code repository for ICCV2021 'Learning to Diversify for Single Domain Generalization'. 

Paper Link: http://arxiv.org/abs/2108.11726

## Update: Single DG with Resnet-18
Recently, we receive increasing enquiry about single DG on PACS with Resnet-18 Backbone. (In the paper, we reported Alexnet result)
Please try hyperparameters lr=0.002 and e=50, to start your experiment. 

We report the following single DG result on PACS, with resnet-18 as the backbone network:

|Src. domain    | P       | A     | C     | S    |avg. |
|---             | ------- |-------|-------| -----| --- |
| Avg. Tar. Acc. | 52.29   | 76.91 | 77.88 | 53.66|65.18|


## Quick start: (Generalizing from art, cartoon, sketch to photo domain with ResNet-18)
1. Install the required packages.
2. Download PACS dataset.
3. Execute the following code.
```
bash run_main_PACS.sh
```

## Change dataset
In line 266-300 of train.py, we provide 3 different datasets settings (PACS, VLCS, OFFICE-HOME).
You can simply uncomment it to start your own experiment. It may require hyper-parameter fine tuning for some of the tasks.


---

# 🔄 Later Addition (Updated Content)

## For DatasetAuditing
**Important Note Regarding Domain Watermark (DW) Implementation**:  
The original code for Domain Watermark from the paper ["Domain Watermark: Effective and Harmless Dataset Copyright Protection Is Closed at Hand" (NeurIPS 2023)](https://arxiv.org/abs/2305.16192) was not publicly released. This code implementation example was developed based on common technical elements between the two papers, following guidance from the original authors. Please note that this serves as a reference implementation and details may differ from the original paper. For any questions regarding the DW method itself, please contact the original authors.

1. Install the required packages.
2. Download imagenet100 dataset.
3. Execute the following code.
bash wm.sh
