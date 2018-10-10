# “Touching to See” and “Seeing to Feel” using Generative Adversarial Networks (GANs)

Implementation of a Conditional Generative Adversarial Network for learning the cross-modal interaction between a visual and tactile perspective. This adapts images from the [ViTac Dataset](https://arxiv.org/pdf/1802.07490.pdf) consisting of fabric materials captured from a camera as a visual perspective and a GelSight sensor as the tactile perspective. we propose a novel framework for the cross-modal sensory data generation for visual and tactile perception. Taking texture perception as an example, we apply conditional generative adversarial networks to generate pseudo visual images or tactile outputs from data of the other modality.

![](https://github.com/SirTune/vis_tac_cross_modal/blob/master/img/cloth_images.png)

Extensive experiments on the ViTac dataset of cloth textures show that the proposed method can produce realistic outputs from other sensory inputs.

Vitac Dataset: [Google Drive](https://drive.google.com/file/d/1uYy4JguBlEeTllF9Ch6ZRixsTprGPpVJ/view?usp=sharing)

___
### CGAN
[Wiki](https://en.wikipedia.org/wiki/Generative_adversarial_network)

GAN: [Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets)

CGAN: [Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014)](https://arxiv.org/abs/1411.1784)

Implementation of a Conditional Generative Adversarial Network. This trains the network to adapt input images from one domain to another. Variables '**in_data_dir**' and '**out_data_dir**' is to be adjusted as the input and output domain respectively.
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
Implementation of several evaluation metrics to evaluate the generated output. Save images in separate folders and adjust variables in the code to locate the folder.
#### Inception Score

[Salimans, Tim, et al. "Improved techniques for training gans." Advances in Neural Information Processing Systems. 2016](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans)

```
cd evaluation
python inception.py
```
#### Structural Similarity Index
Compares two datasets to identify the similarity. Please adjust array in the required order for comparison.

[Wang, Zhou, et al. "Image quality assessment: from error visibility to structural similarity." IEEE transactions on image processing 13.4 (2004): 600-612.](https://ieeexplore.ieee.org/abstract/document/1284395?reload=true)

```
cd evaluation
python ssim.py
```
#### Colour Structural-Similarity Index

[Kolaman, Amir, and Orly Yadid-Pecht. "Quaternion structural similarity: a new quality index for color images." IEEE Transactions on Image Processing 21.4 (2012): 1526-1536](https://ieeexplore.ieee.org/abstract/document/6112222)

Please adjust array in the required order for comparison.
```
cd evaluation
python ssim_colour.py
```
___
### Classification

[Wiki](https://en.wikipedia.org/wiki/AlexNet)

[Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

Applies the alexnet network to test the classification of the images. Save all files in separate folders and generate test files with class labels. To generate text files, adjust variables in 'txt_gen.py' and run:
```
cd alexnet
python txt_gen.py
```
Adjust variables in main.py and run:
```
cd alexnet
python main.py
```
