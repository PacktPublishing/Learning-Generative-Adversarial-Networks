# Learning-Generative-Adversarial-Networks
This is the code repository for [Learning Generative Adversarial Networks](https://www.packtpub.com/big-data-and-business-intelligence/learning-generative-adversarial-networks?utm_source=github&utm_medium=repository&utm_campaign=9781788396417), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Generative models are gaining a lot of popularity among the data scientists, mainly because they facilitate the building of AI systems that consume raw data from a source and automatically builds an understanding of it. Unlike supervised learning methods, generative models do not require labeling of the data which makes it an interesting system to use. This book will help you to build and analyze the deep learning models and apply them to real-world problems. This book will help readers develop intelligent and creative application from a wide variety of dataset (mainly focusing on visual or images).


## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
<div class="packt_code">
nsamples=6
Z_sample = sample_Z(nsamples, noise_dim)
y_sample = np.zeros(shape=[nsamples, num_labels])
y_sample[:, 7] = 1 # generating image based on label
samples = sess.run(G_sample, feed_dict={Z: Z_sample, Y:y_sample})
</div>
```

All of the tools, libraries, and datasets used in this book are open source and
available free of charge. Some cloud environments used in the book offer free trials
for evaluation. With this book, and some adequate exposure to machine learning
(or deep learning), the reader will be able to dive into the creative nature of deep
learning through generative adversarial networks.
You will need to install Python and some additional Python packages using pip to
effectively run the code samples presented in this book.

## Related Products
* [Deep Learning: Practical Neural Networks with Java](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-practical-neural-networks-java?utm_source=github&utm_medium=repository&utm_campaign=9781788470315)

* [Learning Network Penetration Testing with Kali Linux [Video]](https://www.packtpub.com/networking-and-servers/learning-network-penetration-testing-kali-linux-video?utm_source=github&utm_medium=repository&utm_campaign=9781787129481)

* [Learning Docker Networking](https://www.packtpub.com/networking-and-servers/learning-docker-networking?utm_source=github&utm_medium=repository&utm_campaign=9781785280955)

