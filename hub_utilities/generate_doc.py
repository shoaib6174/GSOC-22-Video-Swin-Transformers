"""
Credits: Sayak Paul
"""

import os
from string import Template

import attr

template = Template(
    """# Module $HANDLE
Fine-tunable Video Swin Transformer model pre-trained on the $PRE_TRAIN_DATASET dataset and was then fine-tuned on $FINE_TUNE_DATASET dataset.
<!-- asset-path: https://drive.google.com/drive/folders/1ZVbE5Dd2LxT8lDgNgtziyCB1RfbwzvDd/$ARCHIVE_NAME.tar.gz  -->
<!-- task: video-classification -->
<!-- network-architecture: video-swin-transformer -->
<!-- format: saved_model_2 -->
<!-- fine-tunable: true -->
<!-- colab: https://colab.research.google.com/drive/1McH0gP3UeD_fEMl4MyGM1vbLOFAS-3Vj -->
## Overview
This model is a Video Swin Transformer [1] pre-trained on the $PRE_TRAIN_DATASET dataset and was then fine-tuned on $FINE_TUNE_DATASET dataset. You can find the complete
collection of Swin models on TF-Hub on [this page](https://tfhub.dev/shoaib6174/collections/video-swin).
You can use this model for feature extraction. Please refer to
the Colab Notebook linked on this page for more details.
## Notes
* The original model weights are provided from [2]. They were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.
## References
[1] [Video Swin TransformerZe et al.](https://arxiv.org/abs/2106.13230)
[2] [Video Swin Transformers GitHub](https://github.com/SwinTransformer/Video-Swin-Transformerr)
[3] [GSOC-22-Video-Swin-Transformers GitHub](https://github.com/shoaib6174/GSOC-22-Video-Swin-Transformers)

## Acknowledgements
* [Google Summer of Code 2022](https://summerofcode.withgoogle.com/)
* [Luiz GUStavo Martins](https://www.linkedin.com/in/luiz-gustavo-martins-64ab5891/)
* [Sayak Paul](https://www.linkedin.com/in/sayak-paul/)

"""
)


@attr.s
class Config:
    size = attr.ib(type=str)
    patch_size = attr.ib(type=int)
    window_size = attr.ib(type=int)
    dataset = attr.ib(type=str)
    pre_train_dataset = attr.ib(type=str)
    fine_tune_dataset = attr.ib(type=str)

    type = attr.ib(type=str, default="swin")


    def gcs_folder_name(self):
        
        return f"swin_{self.size}_patch{self.patch_size}_window{self.window_size}_{self.dataset}"
        

    def handle(self):
        return f"shoaib6174/{self.gcs_folder_name()}/1"

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


# swin_base_patch4_window12_384, swin_base_patch4_window12_384_in22k
for c in [
    Config("tiny", 244, 877,  "kinetics400_1k", "ImageNet-1K", "Kinetics 400(1k)"),
    Config("small", 244, 877, "kinetics400_1k", "ImageNet-1K", "Kinetics 400(1k)"),
    Config("base", 244, 1677, "sthv2", "Kinetics 400", "Something-Something V2"),
    Config("base", 244, 877, "kinetics400_1k", "ImageNet-1K", "Kinetics 400(1k)"),
    Config("base", 244, 877, "kinetics400_22k", "ImageNet-22K", "Kinetics 400(22k)"),
    Config("base", 244, 877, "kinetics600_1k", "ImageNet-22K", "Kinetics 600(1k)")]:
    
    pre_train_dataset = c.dataset


    
    print(c)

    save_path = os.path.join(
         "tfhub.dev", c.rel_doc_file_path()
    )
    model_folder = save_path.split("/")[-2]
    model_abs_path = "/".join(save_path.split("/")[:-1])

    if not os.path.exists(model_abs_path):
        os.makedirs(model_abs_path, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                HANDLE=c.handle(),
                PRE_TRAIN_DATASET =c.pre_train_dataset,
                FINE_TUNE_DATASET = c.fine_tune_dataset,
                ARCHIVE_NAME=c.gcs_folder_name(),
            )
        )
