# HappyDerm
Django API to HappyDerm. an app that detects skin cancer in lession pictures using MobileNet and Soft-attention mechanisme.
it achieves an accuracy of 86% by using multiple custom MobileNet model attached to a soft-attention unit. 
the models are trained on the ISIC-2018 dataset to differentiate lesions in a hierarchical manner. 
the first model is binary. and determines the malignancy of the lesion (is it benign or malignant)
according to its decision. the image is then sent to a specialized model the will determine the exact type of the lession.

Beware that the models themselves are not included in this repo. contact me and I will send them to you, email: kai.mehsi@gmail.com

The API uses Django-Rest-Framework and works by posting a multipart form {"image" : img.png} to this url: /api/diag
the server will then return a json with each type of desease and its probability in the image.
Bening types: 
NV = melanocytic nevus, BKL = benign keratosis-like lesions, DF = dermatofibroma.
Malignant types:
AKIEC = actinic keratosis, BCC = basal cell carcinoma, VASC = vascular lesion, MEL = Melanoma.


![Screenshot from 2023-08-16 10-36-37](https://github.com/SoltaneMehdi/HappyDerm/assets/93163687/fae2de8d-10ff-4137-9b3e-28d52b2cdb74)
