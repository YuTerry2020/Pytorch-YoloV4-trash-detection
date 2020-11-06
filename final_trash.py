from ctypes import *
import math
import random
import os
from os import listdir
from os.path import isfile, isdir, join
import cv2
import numpy as np
import datetime
import pandas as pd
import time
# pytorch 
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

from camera import CameraStream
from flask import Flask, render_template, Response
app = Flask(__name__)
app.config["DEBUG"] = True

"""hyper parameters"""
use_cuda = True


# Function to convert   
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1



def YOLO(video_path, filename):

    # global metaMain, netMain, altNames
    trash_set = set()
    license_set = set()
    
    start_time = time.time()
    configPath = "./cfg/yolov4_trash.cfg"
    weightPath = "./backup_trash/yolov4_trash_best.weights"
    metaPath = "./cfg/trash.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    

    m = Darknet(configPath)

    m.print_network()
    m.load_weights(weightPath)
    print('Loading weights from %s... Done!' % (weightPath))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    namesfile = './cfg/trash.names'
    class_names = load_class_names(namesfile)
    fullpath = join(video_path, filename)
    cap = CameraStream(fullpath).start()

    fps = cap.get_fps()      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"   
    size = cap.get_size()
    
    trash_max = 0
    old_time = 0
    # 記錄下來每個丟垃圾時間
    trash_record_dict = {}
    # 拆出時間
    filename_start_time = filename[-10:-4]
    # 分成 hour:minsec
    filename_start_time = filename_start_time[:2] + ":" + filename_start_time[2:]
    # 分成 hour:min:sec
    filename_start_time = filename_start_time[:5] + ":" + filename_start_time[5:]
    filename_tmp_x = time.strptime(filename_start_time.split(',')[0],'%H:%M:%S')
    filename_start_time_s = datetime.timedelta(hours=filename_tmp_x.tm_hour,minutes=filename_tmp_x.tm_min,seconds=filename_tmp_x.tm_sec).total_seconds()

    out = None
    new_size = (416,416)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        old_frame = None
        out_license = False
        out_time = False
        
        if ret:
            old_frame = frame_read
            frame_read = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
       
        
        frame_resized = cv2.resize(frame_read,
                                   new_size,
                                   interpolation=cv2.INTER_LINEAR)

        boxes = do_detect(m, frame_resized, 0.4, 0.6, use_cuda)
        
        trash_center = (0,0)
        license_center = (frame_resized.shape[0], frame_resized.shape[1])
        frame_resized = cv2.resize(old_frame,
                                   new_size,
                                   interpolation=cv2.INTER_LINEAR)
        width = frame_read.shape[1]
        height = frame_read.shape[0]
        for i in range(len(boxes[0])):
            box = boxes[0][i]
            # print(type(box))
            # print(box)
            xmin = int(box[0] * size[0])
            ymin = int(box[1] * size[1])
            xmax = int(box[2] * size[0])
            ymax = int(box[3] * size[1])
            cls_id = box[6]
            # 畫圖
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(frame_resized, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(frame_resized,
                        class_names[cls_id] +
                        " [" + str(round(box[5] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            
            # 確認是否為同個垃圾
            center = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            temp = (center, pt1, pt2)
            similar_trash = False
            similar_license = False        
            if class_names[cls_id] == 'trash':                
                # temp = (center, pt1, pt2)
                if temp not in trash_set:
                    # print('get new trash')
                    # out_time = True
                    for c in trash_set:
                        center_check = abs(c[0][0] - center[0])+ abs(c[0][1] - center[1])
                        if center_check <= 5:
                            similar_trash = True
                            break
                        # 交疊面積超過60%就當作同一個
                        else :
                            if (c[2][0] - xmin) + (c[2][1] - ymin) < (xmax - xmin)+(ymax - ymin) + (c[2][0] - c[1][0])+(c[2][1] - c[1][1]):
                                cross_xmin = max(xmin, c[1][0])
                                cross_ymin = max(ymin, c[1][1])
                                cross_xmax = min(xmax, c[2][0])
                                cross_ymax = min(ymax, c[2][1])
                                cross_area = abs((cross_xmax - cross_xmin)*(cross_ymax - cross_ymin))
                                total_area = (xmax - xmin)*(ymax - ymin) + (c[2][0] - c[1][0])*(c[2][1] - c[1][1]) - cross_area
                                if total_area != 0:
                                    iou = float(cross_area/total_area)
                                    if abs(iou) > 0.6:
                                        similar_trash = True
                                        break
                    
                    if not similar_trash:
                        trash_set.add(temp)
                        out_time = True                         
            else:
                if temp not in license_set:
                    for c in license_set:
                        if abs(c[0][0] - center[0])+abs(c[0][1] - center[1]) <= 8:
                            similar_license = True
                            break
                        else:
                            if (c[2][0] - xmin) + (c[2][1] - ymin) < (xmax - xmin)+(ymax - ymin) + (c[2][0] - c[1][0])+(c[2][1] - c[1][1]):
                                cross_xmin = max(xmin, c[1][0])
                                cross_ymin = max(ymin, c[1][1])
                                cross_xmax = min(xmax, c[2][0])
                                cross_ymax = min(ymax, c[2][1])
                                cross_area = abs((cross_xmax - cross_xmin)*(cross_ymax - cross_ymin))
                                total_area = (xmax - xmin)*(ymax - ymin) + (c[2][0] - c[1][0])*(c[2][1] - c[1][1]) - cross_area
                                if total_area != 0:
                                    iou = float(cross_area/total_area)
                                    if abs(iou) > 0.8:
                                        similar_license = True
                                        break
                    if not similar_license:
                        license_set.add(temp)
                        out_license = True


        result_img = plot_boxes_cv2(frame_resized, boxes[0], savename=None, class_names=class_names)
        frame_num = cap.get_frameNum()
        duration = frame_num/fps
        if out_time and duration > 5:
            if duration > old_time + 10 or old_time == 0:
                trash_max += 1
                old_time = duration
                if out_license:        
                    Drop_trash_time_sec = str(datetime.timedelta(seconds = (filename_start_time_s + duration)))[:8]
                    if trash_max < 2:
                        trash_record_dict = {filename[:-4] +"_Num_" + str(trash_max): [Drop_trash_time_sec, str(1)]}
                    else :
                        trash_record_dict[filename[:-4] +"_Num_" + str(trash_max)] = [Drop_trash_time_sec, str(1)]
                    resized = cv2.resize(result_img,(1280,720), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite('./output/image/' + filename[:-4] + '_' + str(trash_max) + '_1.jpg', resized)
                else:
                    Drop_trash_time_sec = str(datetime.timedelta(seconds = (filename_start_time_s + duration)))[:8]
                    if trash_max < 2:
                        trash_record_dict = {filename[:-4] +"_Num_" + str(trash_max): [Drop_trash_time_sec, str(0)]}
                    else :
                        trash_record_dict[filename[:-4] +"_Num_" + str(trash_max)] = [Drop_trash_time_sec, str(0)]
                    resized = cv2.resize(result_img,(1280,720), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite('./output/image/' + filename[:-4] + '_' + str(trash_max) + '_0.jpg', resized)
                # print('trash_record_dict', trash_record_dict)    
            out_time, out_license = False, False
            
        # cv2.putText(image, "Catch number " + str(trash_max), (10,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [0, 0, 0], 2)
    cap.stop()

    if len(trash_record_dict) > 0:
        Drop_trash_df = pd.DataFrame(list(trash_record_dict.items()), columns=['Filename', 'Record'])
        Drop_trash_df[['RecordTime','Licence']] = pd.DataFrame(Drop_trash_df.Record.tolist(), index= Drop_trash_df.index)
        Drop_trash_df.drop(columns=["Record"], inplace=True)
        print(Drop_trash_df)
        Drop_trash_df.to_csv("./output/csv_file/"+filename[:-4]+".csv", index=False)

        # print('#############start Record Video#################')

        # # filename ='20200927_234800.avi'
        # Trash_df = pd.read_csv("./trash_data/record/csv_file/"+filename[:-4]+".csv")
        # filename_start_time = filename[-10:-4]
        # filename_start_time = filename_start_time[:2] + ":" + filename_start_time[2:]
        # filename_start_time = filename_start_time[:5] + ":" + filename_start_time[5:]
        # filename_tmp_x = time.strptime(filename_start_time.split(',')[0][:-1],'%H:%M:%S')
        # filename_start_time_s = datetime.timedelta(hours=filename_tmp_x.tm_hour,minutes=filename_tmp_x.tm_min,seconds=filename_tmp_x.tm_sec).total_seconds()
        
        # print(Trash_df)

        # for i in range(len(Trash_df)):
        #     fullpath = join(video_path, filename)
        #     video = cv2.VideoCapture(fullpath)
        #     frame_num = 0
        #     frame_width = int(video.get(3)) 
        #     frame_height = int(video.get(4)) 
        
        #     size = (frame_width, frame_height)
        #     result = cv2.VideoWriter('./trash_data/record/video/' + filename[:-4] + '_' + listToString(Trash_df['RecordTime'][i].split(':')) + '_1.avi',  
        #                             cv2.VideoWriter_fourcc(*'MJPG'), 
        #                             10, size) 
        #     x = time.strptime(Trash_df['RecordTime'][i].split(',')[0][:-1],'%H:%M:%S')
        #     y = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        #     y_add_10_sec = y + 10
        #     y_minus_10_sec = y - 10
        #     y_add_10_sec = y_add_10_sec - filename_start_time_s
        #     y_minus_10_sec = y_minus_10_sec - filename_start_time_s

        #     str_add_10_sec = str(datetime.timedelta(seconds=y_add_10_sec))
        #     str_minus_10_sec = str(datetime.timedelta(seconds=y_minus_10_sec))

        #     print('Cuted Video Enging time ', str_add_10_sec, 'Cuted Video Begin time ', str_minus_10_sec)
        #     while(True): 
        #         ret, frame = video.read() 
        #         frame_num += 1
            
        #         if ret == True:  

        #             duration = frame_num/fps
        #             duration_time = str(datetime.timedelta(seconds = (duration)))[:7]

        #             if duration_time >= str_minus_10_sec and duration_time <= str_add_10_sec and frame_num%2 == 0:
        #                 # print('Save....', duration_time)
        #                 result.write(frame) 
                    
        #         else :
        #             print('End ret')
        #             break
        #     video.release() 
        #     result.release()     
        #     # When everything done, release  
        #     # the video capture and video  
        #     # write objects  
            
        #     print("The video was successfully saved")
    else:
        print('Catch Nothing') 
    print('How much time we spend for this video? ' + str(time.time() - start_time))
    print("*****************************End*************************************")

def main():
    path = './input'
    files = listdir(path)
    for f in files:
        fullpath = join(path, f)
        if isfile(fullpath):
            print(f)
            YOLO(path,f)
    situation = True

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')

situation = False
started = False

@app.route("/forward/", methods=['POST'])
def move_forward():
    global started
    forward_message = "程式已經完成了，請拿取檔案"
    #Moving forward code
    if not started:      
        main()
        started = True
    else:
        forward_message = "程式還沒完成"
        
    return render_template('home.html', forward_message=forward_message)

@app.route("/check/", methods=['POST'])
def check():
    global situation
    #Moving forward code
    check_message = "程式開始運行了，請等待"
    if situation:
        check_message = "結束了"
    else:
        check_message = "還在跑請等等"
    return render_template('home.html', check_message=check_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000, threaded=True)
    