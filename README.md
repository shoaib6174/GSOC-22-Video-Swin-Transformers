<p align="center">
<img align="center" src="https://user-images.githubusercontent.com/40586752/194027673-302bce07-e1c9-487c-9d0b-bcb43e35dea2.png">
</p>

# Video Swin Transformers

This summer I participated in [Google Summer of Code 2022](https://summerofcode.withgoogle.com/) as a contributor, and my mentors were   [Luiz GUStavo Martins](https://www.linkedin.com/in/luiz-gustavo-martins-64ab5891/) and  [Sayak Paul](https://www.linkedin.com/in/sayak-paul/).

This repository presents my works on `TensorFlow 2` implementations of the [Video Swin Transformer](https://arxiv.org/abs/2106.13230) models, convertion of `PyTorch` weights to `TensorFlow 2` models and notebook for fine-tuning the models on `UCF101` dataset.

You can find the final report of the project [here](https://github.com/shoaib6174/GSOC-22-Video-Swin-Transformers/blob/main/assets/final_report.md). 


## Outcomes

* A [collection](https://tfhub.dev/) of feature-extractor from `Video Swin Transformer` models on `TensorFlow Hub`
* A [notebook]() for converting `PyTorch` weights to `TensorFlow 2` model for your desired input shape
* Custom `LearningRateSchedule` for [Cosine Decay with WarmUp]()
* A [notebook]() for fine-tuning the  `Video Swin Transformer` backbones using single-gpu after adding `I3D` video classification model as head


