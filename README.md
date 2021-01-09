# Multiple-Rice-Types-Detection
On single image multiple rice types are being detected.

## Table of Contents - 
* [About Project](#about-project)
* [Detailed Explanation about Project](#detailed-explanation-about-project)
* [About Me](#about-me)

## About Project
This project aims for the detecting the multiple rice types in the single image. The rice type taken into consideration are - Basmati, kolam, Idli rice. Each Image has more than 15 rice piece of different types and the photo of the image is taken on dark as well as some lighter background. The Images of the types of Rice is NOT taken from Internet and they are taken by Phone camera. There is bounding box created around each piece of the rice detecting out the correct name of the rice among set of the rices. 

## Detailed Explanation about Project
* First we would link the Google drive with the Google Colab as we are running the code on Google colab for faster processing due to usage of GPU. Also the version used here for tensorflow is 1.15.2

* Then will define the number of training steps - `1000` and number of the evaluation steps - `50`. Thr number of evaluation step is to check the model performance on non train data. Now we would define the model configuration, so I would be using `SSD Mobile Net V2 configuration` - SSD is single shot multibox detector. SSD is thr type of the object detection technique which has reached new records in terms of performance and precision for object detection tasks, scoring over 74% mAP (mean Average Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO. Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network. MultiBox: this is the name of a technique for bounding box regression. Detector: The network is an object detector that also classifies those detected objects. 
  ```
  'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
  }
  ```
  
  TO BE CONTINUE :)
