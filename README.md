# yolov8-application
Integration of yolov8 object model, pose model, visualization function and resnet-18 action classifier.
## Features
1. Integrate the models of yolov8-object and yolov8-pose, and build the pose classifier with ResNet-18.
2. This repo contains the visualization tools for inferencing, such as drawing box and drawing skeleton.
3.  Object detection model and pose estimation model can be called seperately, and note that pose classifier should be called with pose estimation model.
4.  train.py is for training the pose classifier, this classifier can recognize the action of current frame. The input is heatmap-like data of skeleton graph. Data generation can refer to the function transform and heatmap generation in demo.py.
5.  It's recommended that using ImageLoader from torchvision to make dataset for training. The folder format can be refered to [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html). In training dataset, each image represents 1 frame.
## Enviroment
- Linux Ubuntu 20.04
- GPU RTX 4060
- NVIDIA Driver 535.104.05
- CUDA Version 12.2
- PyTorch 2.0.1
- OpenCV 4.8.0
- Numpy 1.26.0
- ultralytics 8.10.184
- sklearn 1.3.0
- Anaconda
## Folder Path
yolov8-application \
 ┣ data \
 ┃ ┗ video.mp4 \
 ┣ model \
 ┃ ┣ classifier.pth \
 ┃ ┣ yolov8-object.pt \
 ┃ ┗ yolov8n-pose.pt \
 ┣ training \
 ┃ ┗ training.py \
 ┣ README.md \
 ┣ demo.py \
 ┗ requirement.txt 
## Step
---
1. Download the repo.
```
git clone https://github.com/nianjingfeng/yolov8-application.git
```
3. Construct conda environment.
```
conda create --name yolov8-application python=3.9
cd yolov8-application
```
3. Install the requirement library.
```
pip install -r requirement.txt
```
4. Download the model from yolov8 website and move to corresponding folder.
5. Modify the name of models and video in demo.py.
6. Start inferencing.
```
python3 demo.py
```
## Citation
---
[ultralytics](https://github.com/ultralytics)