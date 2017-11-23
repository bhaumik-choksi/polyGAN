# polyGAN 

polyGAN is a Generative Adverserial Netowork(GAN) based on the DC-GAN architecture. It is implemented using Keras in Python. polyGAN is capable of training on and generating color images as well as grayscale images of any given resolution. 

It is implemented in Python using the Keras library and tested on MNIST, CIFAR10 and Chars74K image datasets.


## DC-GAN Architecture

![img](http://www.timzhangyuxuan.com/static/images/project_DCGAN/structure.png)


## Results


- Using the MNIST dataset


![img](https://github.com/bhaumik-choksi/polyGAN/blob/master/outputs/mnist/image300.png?raw=true)


- Using the CIFAR10 dataset


![img](https://github.com/bhaumik-choksi/polyGAN/blob/master/outputs/cifar10/image290.png?raw=true)


- Using alphabet images from the Chars74K dataset


![img](https://github.com/bhaumik-choksi/polyGAN/blob/master/outputs/chars74k/image960.png)


## References
[Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).](https://arxiv.org/abs/1511.06434)

## Dependencies

- Python 3.5
- Numpy
- Tensorflow
- Keras
- OpenCV