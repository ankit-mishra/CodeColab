Convolutional Neural Network
----------------------------------------------------------------------------------------
Why Convolution Operation ?
In Image Classification for large images,Matrix will have very large parameters.
It is difficult to get enough data to prevent a neural network from over-fitting.
Computational and memory requirements for training such large number of parameters. 
To solve this problem we implement Convolution operation.
----------------------------------------------------------------------------------------
Padding
When you take 6x6 matrix and convolve with 3x3 matrix, results in 4x4 matrix
Math says if you take n x n matrix and convolve(filter) with f x f filter results in (n-f+1)*(n-f+1)
So, Everytime you apply convolution operator your image shrinks.

Problem : 
1. Image shrinking at every step,it we have hundred layer in a deep network and it shrinks on each layer
then after hundred layer you endup with a layer of very small image.
2. Other is throwing a lot of information from the edges of the images.

To solve both of these problems ,you have to full apply of convolution operation and you can pad the image. 

