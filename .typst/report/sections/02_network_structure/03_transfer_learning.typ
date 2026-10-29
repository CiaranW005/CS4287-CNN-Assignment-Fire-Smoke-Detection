== Transfer Learning

Our ResNet-50 backbone uses pretraiend weights, coming from torchvision's ResNet-50 ImageNet-1K V2. These were trained to classify 1.28 million images into 1,000 classes so our backbone already has knowledge of general features such as edges, texture and shapes. To preserve the pretrained features of the backbone we use a learning rate that is 10 times smaller than the rest of the hybrid model. 

Our YOLOv8 neck and head borrow only the architecture of a YOLO model and uses random weight initialisation, therefore it is trained from scratch.