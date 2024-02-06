# face-detection
The goal is learn how face detection is done using computer vision and also the evolution of this topic into the field of data science and machine learning. Current results show that Haar features are really robust and show good acccuracy for all models (obviously better performance for ensemble techniques compared to single decision tree). But there still requires the need to have a cascade approach for real-time detection of faces in images or video where as maybe CNN does not need to do that. 
TODO: Read documentation and papers whether cascade approach or CNNs are better and which is better for deployment on low resource devices. Also create a real-time face detection using opencv library and generated models. 

### Approach 
- Generate [Haar features](https://github.com/tusharparimi/face-detection/issues/1) for the facial and non-facial datasets.
- Train both [boosting](https://github.com/tusharparimi/face-detection/issues/2) and [bagging](https://github.com/tusharparimi/face-detection/issues/5) based ensemble tree models and compare the performance metrics.

### Performance
- Decision tree:
Mean score of 0.933 with a standard deviation of 0.003

- Random Forest (Bagging):
Mean score of 0.979 with a standard deviation of 0.002

- AdaBoost (Boosting):
Mean score of 0.961 with a standard deviation of 0.002
