from __future__ import print_function
import os
import sys
import subprocess




# Download Fashion MNIST
def download_mnist(dirpath):
 
    if os.path.exists(dirpath):
        print('Found MNIST - skip')
        return
    else:
        os.makedirs(dirpath)
    url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    file_names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(dirpath,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)



if __name__ == '__main__':

    download_mnist('./data/fashion/')
