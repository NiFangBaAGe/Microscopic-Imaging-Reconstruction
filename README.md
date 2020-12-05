# Microscopic-Imaging-Reconstruction
code for "Correction of Out-of-focus Microscopic Images by Deep Learning"

link for "Unnatural L0 Sparse Representation for Natural Image Deblurring" :
http://www.cse.cuhk.edu.hk/~leojia/projects/l0deblur/

link for "Deblurgan: Blind motion deblurring using conditional adversarial networks" :
https://github.com/KupynOrest/DeblurGAN

link for "Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better" :
https://github.com/TAMU-VITA/DeblurGANv2

link for Yolov3 :
https://pjreddie.com/darknet/yolo/

link for dataset :
https://data.mendeley.com/drafts/m3jxgb54c9


# Requirement
Python 3.6
Keras 2.2.4
Tensorflow 1.14。0


# Train
Modify train.py, loading your own dataset, then run
python train.py
to train the model.


# Test
Modify test.py, loading your own dataset and trained model, then run
python test.py
to evaluate the performance.
