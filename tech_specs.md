---
layout: page
title: Tech Specs
permalink: /tech_specs/
---

Time and again, potential customers have asked me to tell give them a more detailed explanation of the skills I've earned over time. I got tired of writing e-mails, so now I'll just list them here, so anybody who is (_that_) interested in the specific technologies I've used can have an in-depth look.


In terms of the of generality / specialization (and, to some degree, of chronology), this is a **bottom-up** presentation of my skills. For a top-down approach, please read the following sub-sections in reverse order.

# 1. Computer Programming
The first time I got paid for the code I wrote was in July 2008 and that's how I've earned my living ever since.

Ok, there was a 1-year long break (2009-2010), during which I only wrote the code for a website, but I was still a student back then and had a job as a computer hardware sales/repair-rep, so don't blame me for it.

Back in 2008 I was writing Java code, but, in the mean time, I've switched to C++ and, later, Python.

## 1.1 C++:
Between July 2010 and August 2016, C++ was the language in which I spent most of the time writing code. Since then, I only use it occasionally, so one could say that my skills are a bit rusty. However, to my surprise, at the time of this writing (October 2019), when having to write a short C++ project for one of my clients, I discovered that I still remember enough of the tricks I used to know so as to produce a decent quality code.

In 2016 when I became self-employed, I had the choice of what language to use, and I've switched to Python. But I still enjoy writing a bit of C++ code from time to time if necessary.

During the various projects I've taken part to I've had the chance to write both low level code for high performance applications and / or hardware platforms with limited resources and high level code shared with lots of other developers, who needed to understand it easily, so that they can modify, use and extend it. 

### Tech
Libraries, language extensions, frameworks, tool chains etc. related to C++

### 1.1.1. `stl`
(including stuff defined by the C++ 11 / 14 standars (I didn't get to C++ 17, though)
- containers: `array`, `map`, `set`, `unordered_map`, `unordered_set`, `vector`, and the lesser known (and widely underapreciated, despite haveing a very efficient `IPP`-based implementation when used compiled with Intel's compiler) `valarray`; I'm also a fan of the Oxford comma
- algorithms: functions the like of which you would find in `<algorithm>` and `<numeric>`. Very interestingly, the `thrust` template library bundled within the CUDA toolkit exposes a very similar API, but is also able to generate CUDA kernels and / or multi-threaded implementations for the operations (under the hood, they would use stuff like `OpenMP` or Intel's Threading Building Blocks (`TBB`) to parallelize the code across available hardware).
- functional programming elements like functors, lambdas, stuff you would find in `<fucntional>` (used e.g., for binding arguments of other functions), especially useful in conjunction with `<algorithm>`, `<numeric>`, and `thrust`, as they allow exposing parallelism opportunities to the runtime / compiler / framework etc.
- smart pointers: mainly the `unique_ptr` and `shared_ptr`, but also `weak_ptr`. Funny story: I've also implemented reference counting myself a couple of times, but would not do it again without relying on the `stl` implementations.
- multi-threading: `<thread>` API and associated stuff like atomics, locks, semaphores, condition variables, futures etc. I've also written some multi-threaded code using the Windows API, `pthread` and, at some point, I needed a fair lock, which I couldn't find an implementation of in the `stl` at the time and used the one in `boost` (or `tbb`?) instead
- other stuff, like `<chrono>` and `<random>`

### 1.1.2. Boost libraries
- most of the functionality I used to use (like, e.g., tuples) has now been (or is in the process of being) standardized in C++11/14/17/..., so you get it in most `stl` implementations bundled with compilers
- `Random`: the mersenne-twister implementation in boost was way better then the one in the stl packaged with the C++ compiler delivered with Visual Studio 2015. This actually made a whole bunch of difference when training some neural nets.
- `Bimap`, `Filesystem`, `Python`, and `Test`

### 1.1.3. Others
- OpenCV (more on this in the Computer Vision section below)
- OpenMP (it's a language extension, not a library, I know...)
- Caffe (everybody used to add their custom layers to caffe at some point)
- Google's `Test` library
- BLAS API implementations (Intel's MKL, OpenBLAS, NVidia's cuBLAS)
- cuDNN
- thrust
- TensorFlow's C API (not the C++ one, because that's not really portable)
- ...

### 1.1.4. C++-related
- a few design patterns (actually, mostly idioms) I remember emerging in my code: pimpl, visitor (via double dispatch), RAII (scope guards), observer, SFINAE, decorator, factory, CAS (copy-and-swap), COW (copy-on-write), nifty counter (aka Schwarz counter), type traits, CRTP, move semantics and rvalue references.
- please bear in mind that I don't remember all the crazy template stuff I used to do, and that I might have to freshen up my knowledge on some of the design patterns I mentioned at the previous point before being able to use them proficiently again.
- for most of my projects, I've usually had to spend a bit of time and effort for ensuring compatibility across various OSs (Windows and Linux), platforms (x86/amd64, ARM), compilers (Microsoft, Intel, GNU, LLVM), IDEs (Microsoft's Visual Studio, JetBrains' CLion, UNIX makefiles - although I'm not very proficient with those, and, a long time ago, Eclipse CDT) and ABIs. To that end, I've done stuff like, e.g., defining COM-compatible APIs for Microsoft Windows dlls, and using `cmake` for builds, static code analysis tools (various linters and checks for standards / coding guidelines) - and, for that matter, dynamic analysis tools for catching heisenbugs and fixing memory leaks (e.g., IBM Rational Purify, Valgrind Memcheck, Microsoft BoundsChecker). Check out [this](https://github.com/bbudescu/hamming) project I wrote for an old interview, for an example.
- I've also had the experience of profiling my code to identify the best opportunities for speedup (candidates for multi-threaded / gpu implementations) using tools like gprof, Callgrind (part of Valgrind), Intel VTune Amplifier, OProfile and with manual instrumentation.

## 1.2. Python
I first used Python sometime around 2012, and it has quickly become my language of choice for computer vision, deep learning and data science applicationsm in the detriment of C++. Since 2016 I've been using it almost exclusively.

Language construct-wise, I've enjoyed using idioms like generators, lambdas, context managers, decorators etc.

**Disclaimer:** Because Python is a very general purpose language, I need to warn you that I have little-to-no experience in stuff other than data science. Like web apps. E.g., I'm able to start a `tornado`/`flask` server for a demo, but I have no idea how to run a `django` site.

### 1.2.1. Libraries
Way too many to build an exhaustive list, but I'll try and cover what I think might be relevant.

#### 1.2.1.1. Notable Built-ins
- `multiprocessing`: in conjunction with `array`, `ctypes` and `numpy` for shared memory buffers (real-time video processing) and syncronization primitives (queues, events, locks, condition variables etc.)
- `asyncio`: not an expert just yet, I've used it in conjuction with `aiohttp`, `aiofiles` and `aiodns`
- `threading`, `collections`, `json`, `pickle`, `csv`, `argparse` and many more...

#### 1.2.1.2. (Somewhat) General Purpose
- `numpy`: in almost all projects have used at least some part of numpy. Being the _Lingua Franca_ when it comes to express data operations (structure definition, storage, access , slicing etc.), no matter what its source and nature is (tabular, images, numeric series etc.)
    - honoroable mention for usefulness and speed: `numpy.linalg`
- `qt`(`PyQt4` / `PyQt5` / `PySide`) for GUIs and, occasionally, real OS threads (working around python's GIL)

#### 1.2.1.3. Deep Learning
`tensorflow` (including the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and `tensorboard`), `keras`, `scikit-learn`, `caffe`, `theano`, `pylearn2` (yes, I'm _that_ old), `hyperopt` (and `hyperas`), `spearmint`

#### 1.2.1.4. Image Processing / Computer Vision
where possible `numpy` (w/ `PIL`/`pillow` for I/O), `OpenCV`, occasionally: `scipy.ndimage`, `mahotas` and `scikit-image` 

#### 1.2.1.5. Visualization
- `matplotlib` (+ `mpld3`), `seaborn`, `plotly`
- also played a bit with: `holoviews`, `bokeh`

#### 1.2.1.6. Statistics
the `statistics` builtin module, `scipy.stats`, `statsmodels`, `pandas`, [`numpy`'s statistics err... section](https://docs.scipy.org/doc/numpy/reference/routines.statistics.html)

#### 1.2.1.7. Optimization
`scipy.optimize`, `openopt`, `cvxpy` and, somewhat related, `autograd`

### 1.2.2. Tools
- virtual environments: `virtualenv`, `venv`, `pipenv`
- `ipython` console
- Jupyter / JupyterLab notebooks
- IDEs: Jetbrains' PyCharm, Microsoft's Visual Studio (when I last used it, it was called PTVS)

## 1.3. MATLAB
I've used MATLAB mostly while doing my assignments for online Coursera lectures, and for a short while, professionally. I liked it a lot and I still can understand code written in this language, but I've given it up for professional projects in favor of Python + numpy + matplotlib

## 1.4. CUDA C/C++
I haven't written a kernel since 2011, but I've used libraries like cudnn, cuBLAS and thrust for implementing operations required  for training and applying neural nets. I think the lowest level (i.e., the closest to the metal of the GPU) code I wrote lately (i.e., cca. 2016) was helping someone else use streams to accelerate their CUDA code for neural nets.

### 1.5. Others
- I can understand code in most C++-based languages. I even used Java professionally a little bit in the beginning of my career (as an intern cca. 2008), and have studied C# in university. So, I get the main idea in code written in Objective C or ActionScript.
- I have a very basic idea of how assembly code works, so I'll be able to understand what some compiler intrinsics interspersed with C code do, but don't ask me to do read / write a full assembly program because it will take an eternity for me to do that.
- I've had a few tentatives (periodically during my career) to use JavaScript professionally, but I've given up on that, since it's too hard for my tiny brain to absorb the relevant information related to all the frameworks that are born, live and die in its ecosystem. However, the basic ideas in the language have stuck, and I'm able to understand (and even write) some simple scripts.
- Languages I can understand (and write) that can be understood by computers, but are not programming languages: Markdown, HTML (basic), LaTeX

# 2. Computer Vision

## 2.1. Old School Image Processing

While being a PhD student, I've been able to learn the fundamentals of classical image processing. Stuff like:
- the pinhole camera model
- how image various image sensors work, e.g., CCD/CMOS chips (I already forgot most of what I knew about these)
- how medical images are acquired (X-Rays, CT and MR scanners, ultrasounds etc.)
- a bit of physics about light and other electromagnetic radiation that can be represented as images
- how the human eye works
- color spaces
- signal processing engineering stuff: interpolation, quantization, sampling etc.
- contrast
- histogram processing (e.g. equalization)
- image filtering with various basic filters
- Fourier Analysis
- morphological image processing
- edge detection
- segmentation methods (e.g., by active contours)
- etc.

All this theory has proven extremely useful over the years. I've been able to apply it time and again in many circumstances. Even if the solution that worked best in the end was based on deep learning, the knowledge about the classical computer vision methods helped me design acquisition systems, perform data normalization and other kinds of preprocessing, establish performance baselines etc.

## 2.2. OpenCV

I've been using OpenCV in almost every project I've worked on ever since I've started doing image processing (2011), in both C++ and Python. In some projects it played a central role (e.g., for the visual odometry projects), while in others it played more of a marginal one (e.g., when training neural nets, I used it mostly for implementing preprocessing operations, e.g. read images, normalization, cropping, augmentation etc.). I'm familiar with (some / most of) the functionality offered by the opencv's `core`, `highgui`, `imgproc`, `video`, `calib3d` modules. Below are the purposes I can remember using it for:
- reading/writing images from/to disk/network/video files
- drawing on images, rotating/resizing/etc. and displaying them on screen
- colorspace conversions, histograms, image filtering (e.g., image denoising, contour detection using morphological, gaussian, laplacian, sobel, gabor etc. filters), distance transforms, thresholding, connected components, segmentation, feature detection (e.g., detect contours using Canny, detect lines using the Hough transform) etc.
- optical flow, background subtraction
- camera calibration, perspective transformations (e.g., image rectification), 3D reconstruction, retrieving camera position and orientation in space using the acquired image

## 2.3. Classical CV projects

Some of the tasks that I solved using only classical computer vision algorithms over time:

### 2.3.1. Blood Vessel Segmentation

I used a morphological active contours algorithm to retrieve the contours of the blood vessels from angiorgraphy imagery. Of course, other filtering was involved e.g., reduce noise, get gradient images, distance transforms, differential geomtery methods for edge detection etc.

### 2.3.2. Document Detection

- camera calibration
- perspective transform for image rectification (reproject the acquired image such that its the document's appearance looks as if the camera's image projection plane was parallel with the document's plane)
- noise filtering using Gaussian and morphological kernels
- edge detection using Canny
- line detection using the Hough transform
- home-baked clustering algorithm to identify line groups that may be potential candidates for document edges, taking into account standard paper aspect ratios and sizes

### 2.3.3. Visual Odometry

- the task was to reconstruct the camera's trajectory within its environment, given a minimal amount of information that would constrain the possible positions
- in this case, the constraint was the prior knowledge of the fixed height at which the camera was situated with respect to the plane of the ground
- to solve the task we used camera calibration, perspective transform, and optical flow algorithms (and, of course, filtering for noise reduction and other \[pre-\|post\]-processing)

#### 2.3.4. 3D Reconstruction (kind of...)
- not sure how to call it, because the term _"3D Reconstruction"_ is generally used to refer to algorithms like SLAM and SfM, and we haven't used those. We actually obtained similar results as you would get from these algorithms, but for a constrained version of the problem.
- Specifically, 3D Reconstruction algorithms like SLAM and SfM aim at creating a 3D map of the environment in which the camera is moving, while _concomitantly_ estimating the camera's trajectory within the (estimated) 3D map reconstruction.
- Our task was a constrained, and hence, easier version of the generic 3D reconstruction task. Namely, we only had to reconstruct  _parts_ of the environment, so as to detect the positions and orientations of specific objects. The rest of the environment was guaranteed to remain unchanged, so we could build a 3D map _a priori_.
- As such, instead of having to _concomitantly_ perform 3D reconstruction and the camera's trajectory estimation within the map, in our case, the two tasks could be solved independently. Conversely, in SLAM / SfM, at each iteration the map is estimated using the current _estimation_ of the camera's trajectory, and, reciprocally, the camera's trajectory is estimated using the current _estimation_ of the map.
- So, we had to:
    - estimate the camera's position and orientation within the environment. We did this by inferring the camera's relative pose (i.e., position and orientation) to objects detected within the acquired images, the positions of which within the environment were known _a priori_.
    - if visible within the acquired images, detect objects of interest and estimate their position and orientation within the environment. If the camera's pose in the environemnt was known from the previous step, the problem reduced to finding the pose of the objects of interest relative to the camera
- Both of the above tasks were further simplified by the fact that the objects we used for orientation were AR markers, which are easy (i.e., efficient, accurate and unambigous) to detect, recognize and reconstruct due to their unique and high-contrast appearance, and due the prior knowledge of their structure that allows for easy identification of their parts.
- In contrast, SLAM and SfM tackle the much harder problem of recognizing arbitrary unknown objects (or parts / regions thereof) across frames that they can base their estimations on.
- One of the challenges we faced during implementation was estimating the camera's position when no landmark was visible. We tackled this problem by:
    - extrapolating current camera pose from previous pose estimations. Actually, in the offline mode, we were able to also use interpolation on both previous and later samples to achieve higher accuracy
    - using multiple cameras and performing fusion of each of their hypotheses about the pose. In the online setting, camera pose estimations were also fused with pose hypotheses based on other sensors (accelerometers, gyroscope, compass, lidar, optical flow-based motion estimators and GPS, where avalable, although, in that particular use case it was mostly unavailable), as well as the commands issued to the actuators and motors that controlled the vehicle carrying the cameras.
- Other challenges included dealing with errors of detection and orientation / reconstruction. We experimented with several optimization algorithms for improving the accuracy of the reconstruction and with methods for improving robustness based on as many measurements as possible, which led to tasks such as outlier detection.

# 3. Deep Learning

I've started studying neural nets in 2012 by enrolling in the first ever class on Coursera: Andrew Ng's excellent lectures on Machine Learning, while I was a PhD student (actually, I had also enrolled in Daphne Kohler's class on Probabilistic Graphical Models, but, although it was quite useful to learn about Bayesian and Markov nets, and gaining a better understanding of the formalism in probability theory, I haven't applied this kind of nets professionally just yet).

I really liked the deep learning approach, so I started an internship where I could further research the subject. During my internship, my main research focus was developing novel algorithms for image segmentation using neural nets, applied on 3D medical imagery (CTs, MRIs etc.). However, my internship's focus was changed to implementing a proprietary C++ library that could train such networks rather than doing scientific research.

After finishing my internship, I worked as a software developer for the same company. During my time as an employee, I've assisted others in using the library I had previously developed during my internship for training and deploying neural networks for various tasks like image {classification, object detection, semantic segmentation}, regression and other tasks related to medical imaging.

As a freelancer I've developed code and trained various networks for my clients that did similar tasks as in my previous job, but on non-medical images. I've applied deep nets on images for tasks such as object {classification, bounding box and keypoint detection, segmentation, recognition} and video enhancement (denoising).

I'm experienced with:
- fully connected and convolutional layers (including implementation details for things like tensor memory layout, strides, dilated filters etc.)
- max / average pooling layers
- various kinds of activation functions: linear, logistic / sigmoid, tanh, ReLU, PReLU, leaky ReLU, softmax, maxout
- normalization layers: batch normalization, local contrast normalization:
- error / loss / cost functions:
    - regression: MSE, RMSE
    - keypoints detection: NME, OKS (and OKS-based mAP)
    - bounding box detection: mAP, Focal Loss
    - classification: binary / categorical cross-entropy, F1 Score
    - segmentation: IoU / Dice / F1 Score
    - recognition: triplet loss, additive angular margin (ArcFace)
    - image enhancement: various perceptual image difference metrics
- regularization: l2 norm (weigh decay), sparsity, \[Spatial\] dropout (binary bernoulli noise) and gaussian noise
- optimization algorithms: \[Mini-batch \| Stochastic\] Gradient Descent, RMSprop, Adagrad, Adadelta \[N\]Adam, Contrastive Divergence
- optimizer variants: momentum, nesterov accelerated gradients
- image preprocessing: standardization, contrast normalization, ZCA whitening etc.
- image augmentation
- architectures: resnet, fpn, unet, mobilenet etc.
- detectors: yolo, SSD, RetinaNet, \[None / Fast / Faster / Mask\]-RCNN heads etc.
- I've used other things as well, but I can't really remember them right now.

I'm familiar with, have not yet used professionally, and would welcome the opportunity to apply:
- recurrent neural nets (+ LSTM)
- recursive neural nets
- GANs
- reinforcement learning
- recommender systems
- RBMs

I'm familiar with, have not yet used professionally, and would not mind applying:
- SVMs
- Random Forests
- Bayesian Networks
- everything Markov: chains, networks, \[hidden\] models

Example of tasks I've tackled over time using deep learning approaches:
- fully connected neural nets for regression of:
    - medical measurements useful for diagnosis of coronary disease
    - toy datasets, e.g. housing prices
- convolutional nets on images for:
    - semantic segmentation of objects (tissues and organs in medical images, skin in natural images)
    - object bounding box detection (e.g., faces, humans)
    - object keypoint detection (facial landmarks, human pose)
    - object classification
    - toy dataset: regression of the angle of rotation between two versions of the same image
    

# 4. Misc.

## 4.1. Docker
Since 2018, all training or inference code I've delivered to my clients is neatly packaged in docker containers, so it can easily be deployed on arbitrary hardware, whether it's our hardware, the client's, or a rented VM in an arbitrary cloud (we've worked with instances on Amazon AWS, Microsoft Azure and Google Compute Platform).

## 4.2. Git
I'm not a `git` guru, but I can find my way around enough to respect a given flow. I've contributed to some open source projects in the past (timidly), and I've had the chance to learn how to follow a strictly enforced policy on committing/testing/integrating the code with larger teams.

## 4.3. Others
- OpenAPI: we've delivered web services that exposed APIs defined using the OpenAPI standard
