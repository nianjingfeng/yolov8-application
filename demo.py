import torch
import torch.nn as nn
import torchvision
import time
import numpy as np
from ultralytics import YOLO
import cv2

#model structure
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 3)
    def forward(self, x):
        x = self.model(x)
        return x

class inference:
    def __init__(self,vid_path,object_detection=True,pose_estimation=True,action_classification=True):
        self.object_detection = object_detection
        self.pose_estimation = pose_estimation
        self.action_classification = action_classification
        if self.object_detection:
            self.yolo_object = YOLO('./model/yolov8_object.pt')
            self.yolo_object.to('cuda')
            self.names = self.yolo_object.names
        if self.pose_estimation:
            self.yolo_pose = YOLO('./model/yolov8n-pose.pt')
            self.yolo_pose.to('cuda')
            self.edges = [[0,0,1,2,3,4,5,5,5,6,6,7,8,11],[1,2,3,4,5,6,6,7,11,8,12,9,10,12]]
        if self.pose_estimation and self.action_classification:
            self.classifier = resnet()
            self.classifier.load_state_dict(torch.load('./model/best_resnet_mix.pth'))
            self.classifier.to('cuda')
            self.classifier.eval()
            self.action_list = ['Assemble','Move','Rest']
        elif self.action_classification and not self.pose_estimation:
            print('Classification model construct failed cause pose estimation model is not loaded')
        self.cap = cv2.VideoCapture(vid_path)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 640, 480)
    def draw_skeleton(self,frame,skeleton,line_width=20,circle_size=25):
        for j in range(len(self.edges[0])):
            cv2.line(frame,(int(skeleton[self.edges[0][j]][0]*frame.shape[1]),int(skeleton[self.edges[0][j]][1]*frame.shape[0])),(int(skeleton[self.edges[1][j]][0]*frame.shape[1]),int(skeleton[self.edges[1][j]][1]*frame.shape[0])),(0,255,0),line_width)
        for j in range(13):
            cv2.circle(frame,(int(skeleton[j][0]*frame.shape[1]),int(skeleton[j][1]*frame.shape[0])), circle_size, (0,0,255), -1)
        return frame
    def draw_box(self,results,frame):
        for result in results:
            for item in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = item
                cls = self.names[int(cls_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                cv2.putText(frame, cls, (int(x1), int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        return frame
    def transform(self,skeleton,width=4056,height=3040):
        for j in range(17):
            skeleton[j][0] = skeleton[j][0] *224 
            skeleton[j][1] = skeleton[j][1] *224*(height/width)
            if j>0:
                skeleton[j][0] = skeleton[j][0] - skeleton[0][0] + 112       
        skeleton[0][0] = 112 
        return skeleton
    def heatmap_generation(self,skeleton):
        img = np.zeros((224,224,3), np.uint8)
        for j in range(13):
            cv2.circle(img,(int(skeleton[j][0]),int(skeleton[j][1])), 5, (0,0,255), -1)
        for j in range(len(self.edges[0])):
            cv2.line(img,(int(skeleton[self.edges[0][j]][0]),int(skeleton[self.edges[0][j]][1])),(int(skeleton[self.edges[1][j]][0]),int(skeleton[self.edges[1][j]][1])),(0,255,0),2)
        return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    def inferencing(self, to_tensor=False):
        while True:
            ret, frame = self.cap.read()
            start = time.time()
            if ret:
                if to_tensor:
                    frame = cv2.resize(frame, (640, 640))
                    frame_input = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
                    frame_input = frame_input/255
                    frame_input = frame_input.to('cuda')
                    width, height = 640,640
                else:
                    frame_input = frame
                    width, height = frame.shape[1],frame.shape[0]
                if self.object_detection:
                    results_object = self.yolo_object(frame_input,conf = 0.6, verbose=False)
                    frame = self.draw_box(results_object,frame)
                if self.pose_estimation:
                    results_pose = self.yolo_pose(frame_input, verbose=False)[0]
                    for result in results_pose:
                        skeleton = result.keypoints.xyn.tolist()[0]
                        frame = self.draw_skeleton(frame, skeleton,3,5) if to_tensor else self.draw_skeleton(frame, skeleton) 
                if self.action_classification:
                    #####select one person
                    skeleton = np.zeros((17,2))
                    for result in results_pose:
                        skeleton = result.keypoints.xyn.tolist()[0] if result.keypoints.xyn.tolist()[0][0][0] > skeleton[0][0] else skeleton
                    transformed_skeleton = self.transform(skeleton,width,height)
                    heatmap = self.heatmap_generation(transformed_skeleton)
                    heatmap = heatmap.to('cuda')
                    result = self.classifier(heatmap)
                    _, predicted = torch.max(result.data, 1)
                    cv2.putText(frame,self.action_list[predicted]+'  fps:'+str(round(1/(time.time()-start),3)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
                else:
                    cv2.putText(frame,' fps:'+str(round(1/(time.time()-start),3)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
                
                #####Available Data: results_object, results_pose, frame, frame_input(torch tensor) 
                #####TODO: Object tracking, Rule, Product Status
                print('fps:',round(1/(time.time()-start),3))
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    self.cap.release()
                    break
            else:  
                cv2.destroyAllWindows()
                self.cap.release()
                break

if __name__ == '__main__':
    model = inference('./data/0904.mp4',True,True,False)
    model.inferencing()