# Bangla-character-recognition-using-deep-neural-network
A Convolutional Deep Neural Network implementation for Bangla character recognition in Pytoch

Dependencies:
Pytorch 0.4
numpy
sklearn
Model Structure:
Model( (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=1600, out_features=400, bias=True)
  (fc2): Linear(in_features=400, out_features=200, bias=True)
  (fc_drop): Dropout(p=0.5)
  (fc3): Linear(in_features=200, out_features=50, bias=True)
  (conv_drop): Dropout(p=0.5)
)
Optimizer used : Adam with initial learning rate 0.001 and weight_decay was 1e-5

Total training time : 2.5hour on CPU
