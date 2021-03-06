# inner-channel
code for the paper Fast Multi-focus Image Fusion using Inner Channel Prior
https://ieeexplore.ieee.org/abstract/document/9530575
## Overview
It is challenging to obtain an image in which all the captured objects are focused, for the finite depth of field (DOF) of photographic lens. Only at a particular distance from the camera can the object be focused and have acceptable sharpness, whereas the sharpness decreases gradually as the object moves away from the sharp focus plane. To obtain an all-in-focus image, we need to fuse the images captured from the same scene at separate object distances. In this paper, we propose a simple but effective sharpness prior—inner channel prior—to detect the focused area of multi-focus images. The inner channel prior is a kind of location feature of natural image sharpness. It is based on a key observation—the high-frequency information of the saturation channel is located inside the object. The inner channel prior indicates the focus degree inside objects, which can obtain better color fidelity and sharper object. Using the proposed prior combined with the multi-scale image fusion, we can directly obtain a sharp image with an extended depth of field focusing on all the objects. Results on a variety of defocused image sequences demonstrate the availability of the proposed prior. In addition, the proposed prior is insensitive to overexposure and can better maintain the color of multi-focus images.
![pic](https://github.com/K-cristina/inner-channel/blob/master/0.png) 
### full procedure 
To see the details of the imaging process of the proposed α-matte boundary defocus model, please reference our paper Fast Multi-focus Image Fusion using Inner Channel Prior
![](https://github.com/K-cristina/inner-channel/blob/master/procedure.png)
### enviroment
The proposed method is performed in visual studio 2017 with OpenCV 3.4.3. OpenCV contrib is required for the guided filter.
