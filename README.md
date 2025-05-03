# KITTI Object data to develop a decision-level fusion perception algorithm
## Introduction
The following is our approach to decision-level fusion; we aim to achieve the integration of 2D and 3D detection results in both image space and point cloud space.
![image](https://github.com/user-attachments/assets/179397ce-1edc-4635-9dc2-39418c09ee89)


## Preparation of 2D and 3D inspection results
You can download our data or prepare it yourself

**1.You can download our data**

our data save in data/object/training,The folder structure is as following:
```
kitti
    object
        training
            calib
               000000.txt
            image_2
               000000.png
            label_2
               000000.txt
            velodyne
               000000.bin
            detect
               000000.txt
            detect3D
               000000.txt
```
**2.You can prepare data yourself**
- "velodyne/image_2/label_2" files save kitti dataset,you can get them from Kitti Object Detection Dataset
- "detect" file saves the 2D detect result,you can detect using your model and save the results to them In the following format
```
type            x1        y1     x2      y2                                 score                                    
Car -1 -1 -10 1133.50 278.19 1225.04 329.51 -1 -1 -1 -1000 -1000 -1000 -10 0.0150    # the format of kitti
#you only need to save the result of(type/x1、x2、y1、y2/score)
```
- "detect3D" file save the 3D detect result,you can still use your model to detect and save the results In the following format
```
Car -1 -1 -10 1133.50 278.19 1225.04 329.51 -1 -1 -1 -1000 -1000 -1000 -10 0.0150    # the format of kitti
```
## Environment preparation
- Our project is based on kitti_object_vis framework
- please follow the link below to configure the environment first
https://github.com/kuixu/kitti_object_vis
**Note:we recommend using conda to prepare environment**

## Demo
- Enter your environment
- Use the code below to demonstrate the fusion result in a 2D plane
```
python test.py 
```
In test.py,you can get many result if you Replace the comments in the main function.
![image](https://github.com/user-attachments/assets/6234fefc-97fe-41aa-910d-98e60f0c2181)
## Demo result 
![image](https://github.com/user-attachments/assets/f3b04b66-0868-4163-a7c6-df8eb3f9dc90)

