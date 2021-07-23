import cv2

class Simplepreprocessor:
    def __init__(self,witdh,height,inter=cv2.INTER_AREA):
        self.witdh =witdh
        self.height = height
        self.inter = inter
    def process(self,image):
        return cv2.resize(image,(self.witdh,self.height),interpolation=self.inter)