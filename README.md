## Pytorch-YoloV4-trash-detection
* This project will detect when a new trash appear in the video.
## Environment
* windows 10
* cuda (choose version by you os and tensorflow version)
* cudnn (choose version by you os and tensorflow version)

## Python library
* look the requirement.txt

## File explain
* final_trash.py: This is the main file, detect function, flask in here
* camera.py: to speed up the FPS and get the frame number
* home.html: sample index page

## Main Goal
* First, run yolov4 in the windows env
* Second, add IOU function and set to decrease the misjudgment times
* Third, speed up the FPS(using Tesla 16G V100, 5min video, 30s finish)
* Forth, use html to be the UI

## Reference
* Download framework
  * git clone https://github.com/Tianxiaomo/pytorch-YOLOv4
