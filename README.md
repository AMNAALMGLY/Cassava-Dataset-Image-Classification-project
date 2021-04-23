# Cassava-Dataset-Image-Classification
![alt text](https://github.com/mohammedElfatihSalah/Cassava-Dataset-Image-Classification/blob/main/diseases.png?raw=true)
The dataset consists of leaf images of the cassava plant, with 9,436 annotated images and 12,595 unlabeled ones.The goal is to learn a model to classify a given image into 4 disease categories or a 5th category indicating a healthy leaf. 🍃 🍂 🍁


# Methodology
## Preprocessing
- A standard Augmentation.
- Cropped image size is 448.

## Model and Hyper-Parameters
- The model is **resnext50**.
- The learning rate is **1e-4**
- The batch size is **20**.
# Results
The model got **91.3%** in public leaderboard.

# Getting started
Just run this command,  
python train.py --fold FOLD_NO --scheduler SCHEDULER --model MODEL
where, 
- MODEL is the model and currently can take the following values,
  - resnext_50 which is [resnext50_32x4d](https://pytorch.org/vision/stable/models.html)
- SCHEDULER is the learning scheduler, and currently can take the following values,
  - cosine_1 which is [CosineAnnealingLR](https://pytorch.org/docs/stable/optim.html?highlight=cosineannealinglr#torch.optim.lr_scheduler.CosineAnnealingLR) 
- FOLD_NO which is the fold number and take the values [0,1,2,3,4].
## Note
before run the above command make sure to get the images from [cassava_images](https://www.kaggle.com/c/cassava-disease/data)  and put in the folder data.
