# MASC - Multi Angle and Scale Convolution
Spin Your Kernel!

The work was published on MICCAI 2021 main conference.



### Instructions

##### The files:

1. Data(folder): The dataset used for demonstration. It contains the patches cropped from CHASE-DB1[1] dataset.
2. model.py: The classes and functions for constructing a MASC model.
3. run.py: The file for loading the dataset and training a MASC-5-2-9 model.

##### How to use them:

1. Download **all the files**.

2. Open run.py and compile it. It was set as a 5-fold cross-validation, so there would be **5 models** to be trained. If you only want one model, make *num_fold=1*.

3. Wait for the training to be completed. Models, examples, and middle outputs can be found in the '/models' folder.

One can visualise kernels with these functions: 
GConv_visual(model): Visualise the hybrid kernels from the 'l3' MASC layer of the input model.
convW_visual(model): Visualise the convolutional parts of the kernels from the 'l3' MASC layer of the input model.
correlation_visual(model): Visualise M matrix of the 'l3' MASC layer.



### Abstract

Many medical and biological applications involve analysing vessel-like structures. Such structures often have no preferred direction and a range of possible scales. We take advantage of this self-similarity by demonstrating a CNN based segmentation system that requires far fewer parameters than conventional approaches. We introduce the Multi Angle and Scale Convolutional Unit (MASC) with a novel training approach called Response Shaping. In particular, by reflecting and rotating a single oriented kernel we can generate four versions at different angles. We show how two basis kernels can lead to the equivalent of eight orientations. This introduces a degree of orientation invariance by construction. We use Gabor functions to guide the training of the kernels, and demonstrate that the resulting kernels generally form rotated versions of the same pattern. Invariance to scale can be added using a pyramid pooling layer. A simple model containing a sequence of five such blocks was tested on CHASE-DB1 dataset, and achieved better performance comparing to the benchmark with only 0.6% of the parameters and 25% of the training examples. The resulting model is fast to compute, converges more rapidly and requires fewer examples to achieve a given performance than more general techniques such as U-Net.



**Trick:** Update key matrix M with momentum(e.g. a=0.99) can effectively improve model performance.

![image-20211015000159145](https://tva1.sinaimg.cn/large/008i3skNgy1gvfmf0fqbmj61g803k74702.jpg)

Paper available at: https://link.springer.com/chapter/10.1007%2F978-3-030-87231-1_57

The preprint version available at: https://www.researchgate.net/publication/354768280_MASC-UnitsTraining_Oriented_Filters_for_Segmenting_Curvilinear_Structures



<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvflc89a8hj60uw0a478n02.jpg" alt="KernelsWeights" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvflc747qdj60us0rh10j02.jpg" alt="scaleRotationIndexMaps" style="zoom:30%;" />




[1] Owen, C.G., Rudnicka, A.R., Mullen, R., Barman, S.A., Monekosso, D., Whincup, P.H., Ng, J. and Paterson, C., 2009. Measuring retinal vessel tortuosity in 10-year-old children: validation of the computer-assisted image analysis of the retina (CAIAR) program. Investigative ophthalmology & visual science, 50(5), pp.2004-2010.
