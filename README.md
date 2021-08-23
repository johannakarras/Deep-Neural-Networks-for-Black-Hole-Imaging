![](Media/header.png "Image Title")

Very long baseline interferometry (VLBI)} uses an array of physically disconnected telescopes to image astronomical objects. In VLBI imaging, a hidden astronomical image is recovered using measurements taken between pairs of telescopes, known as _complex visibilities_. A state-of-the-art approach for VLBI imaging is the _regularized maximum likelihood_ method, which solves for an image that jointly maximizes the measured data log-likelihood and a hand-selected image regularizer.

**We propose  an alternative, data-driven approach that uses a convolutional neural network to reconstruct the hidden image from measurement data.**

<p align="center">
  <img width="600" height="400" src="https://github.com/JSKarras/Deep-Neural-Networks-for-Black-Hole-Imaging/blob/b927df2b473a440e7767d76c2767d1e56eb3149b/Media/Neural%20Network%20Architecture.png" alt="Neural Network Architecture">
</p>

This work was presented as a  [poster](Media/Poster.png) and [extended abstract](Media/Extended_Abstract.pdf) at the WiCV Workshop at CVPR 2021.

## Demos

### Data Download

To test our methods on simulated black hole images, you can download the dataset via [Dropbox](https://www.dropbox.com/s/kv0x5eolg10w52g/bh_sim_data.npy?dl=0).

### RML Demo

You can test our implementation of regularized maximum likelihood method using our [python notebook demo](RML/https://github.com/JSKarras/Deep-Neural-Networks-for-Black-Hole-Imaging/blob/d6d2981685fcfa78ea2f0b88cdb9182fa8a39dd6/RML/RML%20Demo.ipynb).

### Neural Network Demo

First, download our [pretrained model](https://www.dropbox.com/s/fszl68xkdr6hdt6/pretrained_model?dl=0) that was trained using the Fashion MNIST dataset with thermal noises added to the measurement data. 

Next, you can test our neural network reconstruction network with complex visibilities using our [python notebook demo](https://github.com/JSKarras/Deep-Neural-Networks-for-Black-Hole-Imaging/blob/39c2c3a51bfacda9aa8e8527e57967e27dfc8bf8/Neural%20Network/Neural%20Network%20Demo.ipynb).

### Training the Neural Network

You can train your own deep neural network for black hole imaging using our [training script](https://github.com/JSKarras/Deep-Neural-Networks-for-Black-Hole-Imaging/blob/5f61c88d9d05218b4aceffb1c7e54164133f6d51/Neural%20Network/train.py) written in Python and TensorFlow. The model architechture is represented above.
