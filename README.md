# HappyDerm
Django API to HappyDerm. an app that detects skin cancer in lession pictures using MobileNet and Soft-attention mechanisme.
it achieves an accuracy of 86% by using multiple custom MobileNet model attached to a soft-attention unit. 
the models are trained on the ISIC-2018 dataset to differentiate lesions in a hierarchical manner. 
the first model is binary. and determines the malignancy of the lesion (is it benign or malignant)
according to its decision. the image is then sent to a specialized model the will determine the exact type of the lession.

beware that the models themselves are not included in this repo. contact me and I will send them to you, email: kai.mehsi@gmail.com

the API uses Django-Rest-Framework and works by pasting a multipart form {"image" : img.png} to this url: /api/diag
