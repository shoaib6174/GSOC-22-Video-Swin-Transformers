# Final Report

## Overview

The main goal of this project was to make `Video Swin Transformer` availabe in `TensorFlow Hub`. **Video Swin Transformer** was initially described in ["Video Swin Transformer"](https://arxiv.org/abs/2106.13230), which advocates an inductive bias of locality in video Transformers, leading to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization.  This approach achieved state-of-the-art accuracy on a broad range of video recognition benchmarks, including action recognition (`84.9` top-1 accuracy on Kinetics-400 and `86.1` top-1 accuracy on Kinetics-600 with `~20x` less pre-training data and `~3x` smaller model size) and temporal modeling (`69.6` top-1 accuracy on Something-Something v2). In this project, I have created a collection of `Video Swin Transformer` backbone models for `video feature extraction` and published it to `TensorFlow Hub` and provided a colab notebook demontrating  fine-tuning after adding `I3D` video classification model as head.


## Objectives

* Implement  `Video Swin Transformer` in `TensorFlow`
* Convert the pre-trained `PyTorch` weights to `TensorFlow` models and publish them to `TensorFlow Hub`
* Add classification head on top of the backbone model and fine-tune on `UCF101` dataset 



## Milestone Achieved

* Implemented `Video Swin Transformer` model in TF2
* Converted `PyTorch` weights to `TensorFlow` models
* Implemented Cosine Decay with Warmup as custom `LearningRateSchedule` 
* Demostrated fine-tuning


## Work-in-Progress
In fine-tuning, the size of each video is [32,3,224,224] ([frames, channels, height, width]) which ammounts to 4.8 millions elements. Due to the large size of each input, training/fine-tuning on single GPU is not feasible. I am trying to implemnet Distributed training with `MirroredStrategy` for fine-tuning. But I am getting the following error after completation of the first epoch. 

Now, I am trying to figure out how to solve this issue.

## Outcomes

* A [collection](https://tfhub.dev/shoaib6174) of feature-extractor from `Video Swin Transformer` models on `TensorFlow Hub`
* [Fine-tuning Video Swin Transformer with Dummy Inputs](https://colab.research.google.com/drive/1G05XzCNccm9XtMGvYjaeUIliq-z0-Ect) notebook for fine-tuning the  `Video Swin Transformer` backbones after adding `I3D` video classification model as head
* A [notebook](https://colab.research.google.com/drive/1sZIM7_OV1__CFV-WSQguOOZ8VyOsDaGM) for converting `PyTorch` weights to `TensorFlow 2` model for your desired `input shape`
* Custom `LearningRateSchedule` for [Cosine Decay with WarmUp]()

## Acknowledgement 

I would like to thank [Luiz GUStavo Martins](https://www.linkedin.com/in/luiz-gustavo-martins-64ab5891/) and  [Sayak Paul](https://www.linkedin.com/in/sayak-paul/) for their guidance and encouragements. Without their support, I couldn't have reached so far with the project. I would also like to thank [**Google Summer of Code**](https://summerofcode.withgoogle.com) and [**TensorFlow**](https://www.tensorflow.org) for giving me this opportunity.


## Parting thoughts

This project was a great learning experience for me. I got to deep dive into PyTorch and TensorFlow with this project.

In the middle of the project, after converting the PyTorch weights into TensorFlow models, the output of both models for same input weren't matching. My mentors guided me on de-bugging and encouraged me to keep trying. I was stuck with the issue for several weeks but finally I was able to find the issues and solve them. It taught me the importance of patience, persistance and support from the mentors. 

The journey with GSOC'22 was very exciting (and sometimes frustating also). I couldn't have learned so much from elsewhere. I will continue contributing to open-source projects and Tensorflow. 




