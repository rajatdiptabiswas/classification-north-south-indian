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
