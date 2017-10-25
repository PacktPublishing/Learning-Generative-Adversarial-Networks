# MNIST Image generation with Deep Convolutional Generative Adversarial Network (DCGAN) using Keras

git clone https://github.com/jacobgil/keras-dcgan.git

cd keras-dcgan

# Training MNIST
python dcgan.py --mode train  --batch_size 128

# Generating top 5% images according to discriminator
python dcgan.py --mode generate --batch_size 128 --nice


# Implementing Semi-Supervised Learning-SSGAN on cifar-10 using Tensorflow

git clone https://github.com/gitlimlab/SSGAN-Tensorflow.git

cd python download.py --dataset MNIST SVHN CIFAR10

# Download dataset
python download.py --dataset CIFAR10

# Training MNIST
python trainer.py --dataset CIFAR10

# Testing/Evaluating
python evaler.py --dataset CIFAR10 --checkpoint ckpt_dir

# Note: Both the training example procedure will take lot of times if run on cpu. So better to set up tensorflow on gpu and run on it.
