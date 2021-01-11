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
  
* Then I have taken `80:20` train vs test images. With this images, I have generated the XML files and then converted train folder annotation xml files to a single csv file. Same is done with test folder annotation xml files to a single csv file.
    ```
    !python xml_to_csv.py -i data/Images/train -o data/annotations/train_labels.csv -l data/annotations  ---> For Train Images
  	!python xml_to_csv.py -i data/Images/test -o data/annotations/test_labels.csv --------------------------> For Text Images
    ```

* Next is to generate the TFRecords for both train and text images csv files. The TFRecord format is a simple format for storing a sequence of binary records. For future using purpose we need to store up the train record name as train_record_name & test record name as the test_record_name.
    ```
    !python generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --output_path=data/annotations/train.record --img_path=data/Images/train --label_map data/annotations/label_map.pbtxt
    !python generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --output_path=data/annotations/test.record --img_path=data/Images/test --label_map data/annotations/label_map.pbtxt
    ```

* Now to download the Mobilenet SSD v2 Model and then we would be setting TensorFlow pretrained model checkpoint for better proecessing of the model
  ```
  fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
  fine_tune_checkpoint    
  ```
 
* Then we would be Configuring a Training Pipeline, so first we would be joining the pipeline file to already existing folder of `/content/models/research/object_detection/samples/configs/`, giving the name - `pipeline_fname` and then with use of `assert` we would be checking if `pipeline_fname` exist or not. The `assert` statement simply takes input a boolean condition, which when returns true doesn’t return anything, but if it is computed to be false, then it raises an AssertionError along with the optional message provided. So here the error message is printed using `.format()` in python. `str.format()` is one of the string formatting methods in Python3, which allows multiple substitutions and value formatting. This method lets us concatenate elements within a string through positional formatting. Its syntax is `Syntax : { } .format(value)` -> (value) : Can be an integer, floating point numeric constant, string, characters or even variables.
    ```
    import os
  	pipeline_fname = os.path.join('/content/models/research/object_detection/samples/configs/', pipeline_file)
    assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
    ```

* Now we need to define the `labelmap` - In a computer vision dataset, it is common to have annotations referring to class labels. Here for the different types of rice, the class labels include `shapes, colors, size`. In order to annotate an image, an image annotation file will often define the annotations specific to a particular image. In the case where the annotation file does not specify class labels, a label map is referenced to look up the class name. The label map is the separate source of record for class annotations. So first we import the label_map_util - the inbuilt function. Label maps map indices to category names, so that when our convolution network predicts `1`, we know that this corresponds to `Basmati`. So we convert the label map to categories and then we get the category index with the categories and at last we calculate the len of the category indexes as keys. 
    ```
    def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
    ```

* Now to start the training - We will do `model_dir = 'training/'` and if we want to remove the content if it is already present in the output directory i.e invlaid data then to use `!rm -rf {model_dir}`. Now we will make the directory - This means, `os.makedirs()` method in Python is used to create a directory recursively. That means while making leaf directory if any intermediate-level directory is missing, os.makedirs() method will create them all. It defines the `path` and `directory exist or not`. Its syntax is `Syntax: os.makedirs(path, exist_ok = False)` - Path: A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path. exist_ok (optional) : A default value `False` is used for this parameter. If the target directory already exists, if its value is false then OSError is raised otherwise not. For value True, it leaves target directory unaltered.
    ```
    model_dir = 'training/'
    !rm -rf {model_dir}
    os.makedirs(model_dir, exist_ok=True)
    ```

* Now comes the main part of training the model. we define `pipeline_config_path`, `model_dir`, `number of training steps`, `number of Evaluation steps` and `logtostderr`. As name suggest `logtostderr` sends the logs to STDERR i.e standard error file. This means : Every command could send it's output to one of two places: a) it could be valid output or b) it could be an error message. It does the same with the errors as it does with the standard output; it sends them directly to your terminal screen if their are any. 
    ```
    !python /content/models/research/object_detection/model_main.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --num_eval_steps={num_eval_steps}
    ```
