# Classification of Indians into North and South Indians
Comparing the accuracies of different convolutional neural networks for the task of classifying Indians into North and South Indians using facial image data

*This project was made for the **CS-1651 Mini Project** course at **MNNIT Allahabad***

*The documents made for the project can be found in the `docs` branch*

## Dataset description

The first course of action was to collect images of North Indian and South Indian faces. Used Google Custom Search API to get images using common regional surnames. This resulted in images that were popular on the internet and did not provide us with images of the average civilian.

This posed a problem. Fortunately, came across [CNSIFD](https://github.com/harish2006/CNSIFD) *(Centre for Neuroscience Indian Face Dataset)*. This contained 500 x 500 gray scale images with normalised faces in elliptical cropped window.

The dataset provided two `.mat` files that were of interest to us.

- `cnsifd_imgs.mat`
- `cnsifd_info.mat`

*`cnsifd_imgs.mat` contained all the face image data and `cnsifd_info.mat` contained all the labels for the face images and any additional information*

The image files extracted from `cnsifd_imgs.mat` in `.csv` format. There are 1647 `.csv` files in total. Each of the `.csv` files are image files of around 500 x 350 dimensions. Used the `imshow` function in Octave to check whether they really are images. Each `.csv` matrix will produce an image that is pre-processed (black and white, elliptical).

For example, the first 10 image files have the following dimensions.

```octave
octave:32> for i = 1:10
\> size(cnsifd_imgs{1,i})
\> end
              ans =
                 501   380
              ans =
                 501   356
              ans =
                 501   350
              ans =
                 501   353
              ans =
                 501   382
              ans =
                 501   368
              ans =
                 501   371
              ans =
                 501   389
              ans =
                 501   368
              ans =
                 501   350
```

Each of the face data matrices have been converted to `.csv` and stored in the `cnsifd/cnsifd-imgs` folder.

The `cnsifd_info.mat` file had the following fields.

```
fields =
   {
      [1,1] = source_dataset: nfaces x 1, 1-SET1, 2-SET2
      [2,1] = region: nfaces x 1, 1-north, 0-south
      [3,1] = gender: nfaces x 1, 1, 1-male, 0-female
      [4,1] = age: nfaces x 1 declared age in years or nan if not declared
      [5,1] = weight: nfaces x 1 declared weight in kg or nan if not declared
      [6,1] = height: nfaces x 1 declared height in cms or nan if not declared
      [7,1] = pc: nfaces x 1 percentage correct in a north/south categorisation task, nan if not declared
      [8,1] = landmarks: 76 x 2, x,y coordinates of aam landmarks
      [9,1] = landmarksm: 80 x 2, x,y coordinates of aam landmarks
      [10,1] = intensity_landmarks: 31 x 3 landmark ids for face patches
      [11,1] = spatial_landmarks: 32 x 3 landmark ids for face distance measurements
      [12,1] = spatial: nfaces x 23 , spatial measurements
      [13,1] = intensity: nfaces x 31 
      [14,1] = [](0x0)
      [15,1] = [](0x0)
      [16,1] = [](0x0)
      [17,1] = [](0x0)
      [18,1] = [](0x0)
      [19,1] = allfeatures : nfaces x 1446 matrix of all features together in order spatial, intensity, spatial_ratio and intensity_ratio
      [20,1] = allfeatures_raw : nfaces x 1446 normalised matrix of all features together in order spatial, intensity, spatial_ratio and intensity_ratio
      [21,1] = bf.bflandmarks: nfaces x ?  x 2 selected aam landmarks for each face
      [22,1] = bf.bftriangles: nfaces x ? x 3 x 2 x,y coordinates of triangle vertices
      [23,1] = bf.bfspatial: nfaces x ?  measurements between landmarks
      [24,1] = bf.bfintensity: nfaces x ?  measurements on triangles on faces
      [25,1] = cnn: three fields having CNN-F, CNN-A, CNN-G features
      [26,1] = moments: nfaces x 7 intensity moments
      [27,1] = lbp: nfaces x 1328 local binary patterns
      [28,1] = hog: nfaces x 6723 histograms of gradients at multiple scales
      [29,1] = siex: nfaces x 1647 exhaustive measurements from triangulating the face
      [30,1] = spatial_ratio: nfaces x 231 , spatial ratio measurements
      [31,1] = spatial_product: nfaces x 231 , spatial product measurements
      [32,1] = intensity_ratio: nfaces x 465 , intensity ratio measurements
      [33,1] = intensity_product: nfaces x 465 , intensity product measurements
   }
```


The first 7 nfaces x 1 data was used to make the `info.csv` file. The headings in the `.csv` were manually added in for additional reference.

<img width="100%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/data-info/dataset-info.png">
<p align="center">
  <em> Summary of the face dataset </em>
</p>

<p align="center">
  <img width="49%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/data-info/age-dist.jpg"> <img width="49%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/data-info/data-dist.png">
</p>
<p align="center">
  <em> Age and gender distribution of the samples </em>
</p>

<p align="center">
  <img width="40%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/data-info/dataset-region-gender.png">
</p>
<p align="center">
  <em> Gender distribution table </em>
</p>

Once the data from the CNSIFD dataset was converted to `.csv`, the rows that did not include a label (i.e. images with the label as `NaN`) were removed from consideration. The `csv` files were updated to remove those rows and the indices were updated accordingly.
Using the updated `.csv`, the required image `csv`s were converted to `.png` images using `matplotlib.pyplot.imsave`.

### Images obtained from the dataset
<img width="100%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/output/faces-north.png">
<img width="100%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/output/faces-south.png">

The images were then separated into training, test and cross validation sets using the [`split_folders`](https://github.com/jfilter/split-folders) library.
<img width="30%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/data-info/dataset-size.png">

## Convolutional Neural Networks used

### AlexNet
AlexNet is a convolutional neural network that is  8 layers deep and can classify images into 1000 object categories. It is composed of 5 convolutional layers followed by 3 fully connected layers. AlexNet, proposed by Alex Krizhevsky, uses ReLu, instead of a tanh or sigmoid function which was the earlier standard for traditional neural networks. The advantage of ReLu over the sigmoid function is that it trains much faster than the latter. This is due to the fact that the derivative of sigmoid becomes very small in the saturating region and therefore the updates to the weights almost vanish. Another problem that this architecture solved was that it reduced overfitting by using a Dropout layer after every FC layer.
<p align="center">
  <img width="100%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/neural-network-architectures/alexnet-arch.png">
</p> 
<p align="center">
  <em> AlexNet neural network structure </em>
</p>

### VGG
VGG16 is a convolutional neural network model proposed by K. Simonyan and A.Zisserman from the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". VGG19 has a similar model architecture as VGG16 with three additional convolutional layers, it consists of a total of 16 Convolution layers and 3 dense layers.
<p align="center">
  <img width="80%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/neural-network-architectures/vgg16.png">
</p> 
<p align="center">
  <em> VGG16 architecture </em>
</p>
<p align="center">
  <img width="80%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/neural-network-architectures/vgg16-neural-network.jpg">
</p> 
<p align="center">
  <em> VGG16 neural network structure </em>
</p>
<p align="center">
  <img width="85%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/neural-network-architectures/vgg19.jpeg">
</p> 
<p align="center">
  <em> VGG19 neural network structure </em>
</p>

### ResNet
Every other previous model before the ResNet used deep neural networks in which many convolutional layers were stacked one after the other. It was believed that deeper networks perform better. However, it turned out that this was not really true. 

Deep networks face the following problems
- network becomes difficult to optimize  
- vanishing/exploding gradients
- degradation problem (accuracy first saturates and then degrades)

To address these problem, authors of the ResNet architecture came up with the idea of skip connections with the hypothesis that the deeper layers should be able to learn something as equal as shallower layers. A possible solution was to copy the activations from shallower layers and setting additional layers to identity mapping. These connections were enabled by skip connections.

<p align="center">
  <img width="35%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/neural-network-architectures/resnet.png">
</p>
<p align="center">
  <em> ResNet neural network structure </em>
</p>

## Results

<p align="center">
  <img width="19%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/confusion-matrices/cm_alexnet.png"> <img width="19%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/confusion-matrices/cm_vgg16.png"> <img width="19%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/confusion-matrices/cm_vgg19.png"> <img width="19%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/confusion-matrices/cm_resnet50.png"> <img width="19%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/confusion-matrices/cm_resnet152.png">
</p>
<p align="center">
  <b>Confusion Matrices</b> (left to right): <em> AlexNet; VGG16; VGG19; ResNet50; ResNet152 </em>
</p>

<p align="center">
  <img width="75%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/result/confusion.png">
</p>
<p align="center">
  <em> Confusion bar chart </em>
</p>

<p align="center">
  <img width="75%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/result/accuracy.png">
</p>
<p align="center">
  <em> Accuracy bar chart </em>
</p>

<p align="center">
  <img width="75%" src="https://github.com/rajatdiptabiswas/classification-north-south-indian/blob/images/result/train-vs-valid-loss.png">
</p>
<p align="center">
  <em> Training set vs validation set bar chart </em>
</p>

## Built With

* [Google Colab](https://colab.research.google.com/) - Research tool for machine learning education and research
* [JupyterLab](https://github.com/jupyterlab/jupyterlab) - Next-generation web-based user interface for Project Jupyter
* [Octave](https://www.gnu.org/software/octave/) - Open-source software featuring a high-level programming language, primarily intended for numerical computations
* [MATLAB](https://www.mathworks.com/products/matlab.html) - Multi-paradigm numerical computing environment and proprietary programming language developed by MathWorks
* [fast.ai](https://www.fast.ai) - Easy to use deep learning library
* [PyTorch](https://pytorch.org/) - Open-source machine learning library for Python, based on Torch
* [Keras](https://keras.io/) - Open-source neural-network library written in Python, capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, or PlaidML
* [pandas](https://pandas.pydata.org/) - Software library written for the Python programming language for data manipulation and analysis
* [Matplotlib](https://matplotlib.org/) - Plotting library for the Python programming language
* [NumPy](https://www.numpy.org/) - Library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* [SciPy](https://www.scipy.org/) - Free and open-source Python library used for scientific computing and technical computing
* [seaborn](https://seaborn.pydata.org/) - A Python data visualization library based on matplotlib
* [Split Folders](https://github.com/jfilter/split-folders) - Automatically split folders with files (i.e. images) into training, validation and test (dataset) folders
* [Google Sheets](https://docs.google.com/spreadsheets/u/0/) - Spreadsheet program included as part of a free, web-based software office suite offered by Google within its Google Drive service
* [Google Drive](https://www.google.com/drive/) - File storage and synchronization service developed by Google

## Authors

* **Rajat Dipta Biswas** - [rajatdiptabiswas](https://github.com/rajatdiptabiswas)
* **Tuhin Subhra Patra** - [armag-pro](https://github.com/armag-pro)
* **S Pranav Ganesh**
* **Upmanyu Jamwal**

See also the list of [contributors](https://github.com/rajatdiptabiswas/GeoARgraphy/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* fast.ai course | [Image Classification](http://course.fast.ai/videos/?lesson=1)
* Coursera | [Convolutional Neural Networks by Andrew Ng](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
* Documentations - [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/), [Octave](https://octave.org/doc/interpreter/), [MATLAB](https://www.mathworks.com/help/matlab/), [Matplotlib](https://matplotlib.org/contents.html), [PyTorch](https://pytorch.org/docs/stable/index.html), [Keras](https://keras.io), [NumPy](https://docs.scipy.org/doc/numpy-1.13.0/reference/), [SciPy](https://docs.scipy.org/doc/scipy/reference/), [pandas](https://pandas.pydata.org/pandas-docs/stable/), [seaborn](https://seaborn.pydata.org/api.html)
* [StackOverflow](https://stackoverflow.com)
