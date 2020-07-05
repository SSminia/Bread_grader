import numpy
import os
import cv2
from utils.utils import *
from models import *
from utils.datasets import *

class Yolodetect(object):

    def __init__(self,config):
        if config != None:
            self.config_dict = config
        else:
            self.config_dict = {
                "cfg": "cfg/yolov3.cfg",
                "names": "data/names.txt",
                "weights": "weights/best.pt",
                "conf_thres": 0.1,
                "img-size": 512,
                "iou-thres": 0.7
            }

    def loadimg(self, path):
          # Read image
        if os.path.exists(path): 
            img0 = cv2.imread(path)  # BGR
            img = letterbox(img0, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            return (path, img, img0)
        else:
            return None


    def startmodel(self):
        self.imgsz = self.config_dict["img-size"]  
        self.weights = self.config_dict["weights"]

        # Initialize
        self.device = torch_utils.select_device(device='')

        # Initialize model
        self.model = Darknet(self.config_dict["cfg"], self.imgsz)

        # Load weights
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        # Eval mode
        self.model.to(self.device).eval()

        # Get names and colors
        self.names = load_classes(self.config_dict["names"])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def findresult(self, path, img1, im0s):
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if False else img.float()) if self.device.type != 'cpu' else None  # run once
        # path, img, im0s = self.loadimg(imgsource)
        img = img1
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)       

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.config_dict["conf_thres"], self.config_dict["iou-thres"], multi_label=False, classes=None, agnostic=False)
        returnval = [] 
            # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p = path
            s = ''
            im0 = im0s.copy()

            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from self.imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                for *xyxy, conf, cls in det: 
                    roi = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    returnval.append([roi, float(conf), self.names[int(cls)]])        
                               

                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    # plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
                    # cv2.imshow("bread", im0)
                    # cv2.waitKey(0)

        return returnval

class Crop(object):
    def __init__(self):
        pass

    def newimg(self, sorceimg, roi): # cropping the image based on detection
        maxh, maxw, _= sorceimg.shape

        x = roi[0]-int(roi[0]*0.15) #starting point
        x = max(x, 0) 
        y = roi[1]-int(roi[1]*0.15)
        y = max(y, 0) 
        h = (roi[3]-roi[1])*1.5
        h = int(min(h, maxh-y))
        w = (roi[2]-roi[0])*1.5
        w = int(min(w, maxw-x))
        
        croppedimg = sorceimg[y:y+h, x:x+w]
       # cv2.imshow("croppedimg", croppedimg)
        #print(croppedimg.shape)
        #cv2.waitKey(0)
        cv2.imwrite("placeholder.png", croppedimg)

class Grade(object): 
    def __init__(self):
        pass

    def grading(self, Iarray):
        A = []
        B = []
        for  x in Iarray:
            A.append(x[1])
            B.append(x[2])
        positive = 0.0
        negative = 0.0
        for x in range (0, len(A)):
            if B[x] == "baked":
                positive = positive + A[x]
            else:
                negative = negative + A[x]
        if positive == 0:
            positive = 1
        if negative == 0:
            negative = 0.5 * (1- positive)
        w = (positive-negative) / positive
        print (w) 
        grader = ["A","B","C","D","E","F"]
        t=1
        i = 0
        grade = ""
        while w <= t:
            if w < (t - 0.15):
                t = t- 0.15
            else:   
                grade = grader[i]
                break
            i = i + 1
            if i > 5:
                grade = grader[5]
                break
        return grade
        




if __name__ == '__main__':
    test = Yolodetect(None)
    test.startmodel()
    d = 0
    image_array = ["D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.01.41 (4).jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.01.43 (3).jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.01.44.jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.24.17.jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.24.20.jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.24.27 (3).jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.24.28 (4).jpeg",
                    "D:\minor\Vision\Vision\Bread\model test\WhatsApp Image 2020-07-02 at 19.24.29 (1).jpeg"]
    for pic in image_array:
        p,i1,i0 = test.loadimg(pic)
        potato = test.findresult(p,i1,i0)
        print(potato)
        chop = Crop()

        for x in potato: 
            placeholderimg = chop.newimg(i0, x[0])
            pp,pi1,pi0 = test.loadimg("placeholder.png")
            potato2 = test.findresult(pp,pi1,pi0)
        #  print(potato2)
        rat = Grade()
        value = rat.grading(potato)
        print (value)
        maxh, maxw, _= i0.shape
        cv2.putText(i0, value,(0, int(maxh - maxh * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 2)
        #cv2.imshow("w windwos", i0)
        d =d + 1
        cv2.imwrite("windows" + str(d) + ".png", i0)
    # cv2.waitKey(0)
        
        
    
    # potat1 array [roi, confidence, class]
                #roi, confidence, class
    #potato2 after crop


