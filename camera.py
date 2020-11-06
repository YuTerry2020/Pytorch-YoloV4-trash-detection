from threading import Thread, Lock
import cv2


class CameraStream(object):
    def __init__(self, filename, src=0):
        #self.stream = cv2.VideoCapture('highway_car_tracking.mp4')
        self.stream = cv2.VideoCapture(filename)
        self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #self.stream.set(3,640) #set frame width
        #self.stream.set(4,480) #set frame height
        #self.stream.set(cv2.CAP_PROP_FPS, 10) #adjusting fps to 5
        (self.grabbed, self.frame) = self.stream.read()
        self.frameNum = 1
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def get_fps(self):
        return self.fps
    
    def get_size(self):
        return self.size

    def get_frameNum(self):
        return self.frameNum

    def update(self):
        while self.started:
            #print('update_1')
            (self.grabbed, self.frame) = self.stream.read()
            self.frameNum += 1
            #print('update_2')
            # self.read_lock.acquire()
            # self.grabbed, self.frame = grabbed, frame
            # self.read_lock.release()
            #print('------update_5----------')
        

    def read(self):
        # #print('read_1')
        # self.read_lock.acquire()
        # #print('read_2')
        # grab, frame =  self.grabbed, self.frame.copy()
        # #print('read_3')
        # self.read_lock.release()
        return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()