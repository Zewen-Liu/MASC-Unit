# MASC: Spin Your Kernel!
 **Original title**: MASC - Multi Angle and Scale Convolution

The work was published on MICCAI 2021 main conference.

##### Abstract

Many medical and biological applications involve analysing vessel-like structures. Such structures often have no preferred direction and a range of possible scales. We take advantage of this self-similarity by demonstrating a CNN based segmentation system that requires far fewer parameters than conventional approaches. We introduce the Multi Angle and Scale Convolutional Unit (MASC) with a novel training approach called Response Shaping. In particular, by reflecting and rotating a single oriented kernel we can generate four versions at different angles. We show how two basis kernels can lead to the equivalent of eight orientations. This introduces a degree of orientation invariance by construction. We use Gabor functions to guide the training of the kernels, and demonstrate that the resulting kernels generally form rotated versions of the same pattern. Invariance to scale can be added using a pyramid pooling layer. A simple model containing a sequence of five such blocks was tested on CHASE-DB1 dataset, and achieved better performance comparing to the benchmark with only 0.6% of the parameters and 25% of the training examples. The resulting model is fast to compute, converges more rapidly and requires fewer examples to achieve a given performance than more general techniques such as U-Net.



Paper available at: https://link.springer.com/chapter/10.1007%2F978-3-030-87231-1_57

The preprint version available at: https://www.researchgate.net/publication/354768280_MASC-UnitsTraining_Oriented_Filters_for_Segmenting_Curvilinear_Structures



<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvflc89a8hj60uw0a478n02.jpg" alt="KernelsWeights" style="zoom:67%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvflc747qdj60us0rh10j02.jpg" alt="scaleRotationIndexMaps" style="zoom:50%;" />
