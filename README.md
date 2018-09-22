# “Touching to See” and “Seeing to Feel” using Generative Adversarial Networks (GANs)

Implementation of a Conditional Generative Adversarial Network for learning the cross-modal interaction between a visual and tactile persepective. This adapts images from the [ViTac Dataset](https://arxiv.org/pdf/1802.07490.pdf) consisting of fabric materials captured from a camera as a visual perspectiva and a GelSight sensor as the tactile perspective. we propose a novel framework for the cross-modal sensory data generation for visual and tactile perception. Taking texture perception as an example, we apply conditional generative adversarial networks to generate pseudo visual images or tactile outputs from data of the other modality.

![](https://github.com/SirTune/vis_tac_cross_modal/blob/master/img/cloth_images.png)

Extensive experiments on the ViTac dataset of cloth textures show that the proposed method can produce realistic outputs from other sensory inputs.
___
### CGAN
Implementation of a Conditional Generative Adversarial Network. This trains the network to adapt input images from one domain to another. Variables '**in_data_dir**' and '**out_data_dir**' is to be adjusted as the input and output domain respectivly.
Current layers only allow 256x256 images, use ./images/resize.py to resize the input images.

Dependencies:
* Python 3.7.0
* Tensorflow 0.12.0
* Numpy 1.15.0
* Scipy 1.1.0

```
cd cgan
python cgan.py --mode train
```

___
### Evaluation
Implementation of several evaluation metrics to evalute the generated output
#### Inception Score

#### Structural Similarity Index


#### Colour Structural-Similarity Index

___
### Classification
