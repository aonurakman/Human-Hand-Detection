> cfg/: COnfiguration for YOLO, for detection.
> GT_boxes/: GT boxes provided with test data.
> in_imgs/: Test set.
> out_boxes/: Output bounding boxes in .txt format. One .txt file for each input. Order: [X] [Y] [W] [H], one line for each detection.
> out_imgs/: Generated bounding boxes represented visually.
> weights/: Weights used and unused for the detection.
> _log_.txt: Log during runtime. Contains IoU scores for each image.
> yolo_detection.py: COde for estimation. Run for obtaining output yourself.
