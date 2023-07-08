# Human Hand Detection & Segmentation 

Team members: **Onur Akman**, **Henrik Adrian Hansen**, **Matthias Schmitz**

## Phase 1
Detecting bounding boxes surrounding the human hand in the given input image. Used transfer learning with YOLOv3. Fully developed with Python.
Output is constructed by fusing the results of multiple weights, and then OpenCV's non-maximum suppression. Each one of the weights used in this step is from the same architecture (YOLOv3) but fine-tuned on a different dataset. This enabled us to use the strengths of each one of them and make up for their weaknesses.

![Output of Phase 1](https://i.hizliresim.com/a3zydv5.PNG)

## Phase 2

Bounding boxes detected in Phase 1 are the input for Phase 2. This phase is completely developed with C++, using methods built-in OpenCV. The output will be pixel-wise segmentations of the hand regions in the given image. A pipeline is created using multiple techniques, and these are first and second derivative filtering, thresholds, contour finding and filling, segmentation by color (RGB) and the GrabCut algorithm.

![Output of Phase 2](https://i.hizliresim.com/s04m5om.PNG)

## Report
More details about the implementation, the results, and the evaluations.
