# Progress CNN

- apply code of exercise -> worked well, but maybe the model is too large
- mean and std computed for each fold
- slow training :
  - efficient data loading
  - mixed precision training
  - try to reduce the model to minimal size -> worked well
- try to reduce the image size, worked well
- try to reduce the image from 768x768 to 5x5 ->worked well, how is possible? :
  Is learning the color of the class?
- remove the background :
  - still learning the color of the class
- gray scale : hard to classify :
  - resnet style model is working well
