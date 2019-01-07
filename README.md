

# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/center_lane.jpg "Center lane driving"
[image2]: ./examples/recovery_1.jpg "Recovery 1"
[image3]: ./examples/recovery_2.jpg "Recovery 2"
[image4]: ./examples/recovery_3.jpg "Recovery 3"
[image5]: ./examples/normal.jpg "Normal"
[image6]: ./examples/reversed.jpg "Flipped"
[image7]: ./examples/stats.jpg "Stats"


## Rubric Points
### Here I will consider the [rubric points]
(https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 file containing the video created using the autonomous recording mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of three convolution layers and four dense layers and structure as given below (Refer 106 to 127)

Convolution - 4 layers and 5x5 filter size
Max Pooling - 2x2 filter size

Convolution - 8 layers and 5x5 filter size
Max Pooling - 2x2 filter size

Convolution - 16 layers and 3x3 filter size

Flatten Layer
Dense 800
Dense 200
Dense 10
Dense 1

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 107). 

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py - lines 120 and 123). 

The model was trained and validated on different data sets to ensure that the model was not over fitting (code line 129-132). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (refer model.py line 129).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to analyze the training data given and to look at the car track given for testing.

My first step was to use a convolution neural network model similar to the modified **LeNet** model.  I thought this model might be appropriate because it gets trained pretty fast and accuracy is better than other similar models.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over fitting. 

To combat the over fitting, I modified the model so that it has two drop out layers in between the fully connected layers. After adding dropouts, the model performed very well and over fitting was greatly reduced

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like in mud road after the bridge. Also car went off road when there was a sharp turn while finishing the lap. To improve the driving behavior in these cases, I collected data from the specific turns in the road for a few more times and it improved the driving performance near those areas

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (Refer 106 to 127) consisted of a convolution neural network with the following layers and layer sizes:

Convolution - 4 layers and 5x5 filter size
Max Pooling - 2x2 filter size
Convolution - 8 layers and 5x5 filter size
Max Pooling - 2x2 filter size
Convolution - 16 layers and 3x3 filter size
Flatten Layer
Dense 800
Dense 200
Dense 10
Dense 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to  These images below show what a recovery looks like starting from :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the car to learn driving in the opposite direction also. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 24108 number of data points. 

Initial sample size : 24108
Left angle : 2811
Center angle : 17769
Right angle : 3528

**Preprocessing**

It can be seen than center angle data is way higher than the left angle and right angle data. Hence I removed excessive the center angle data.

Here is a bar chart visualization showing the before and after preprocessing process

![alt text][image7]

After rejecting the redundant center angle data total sample size became 10781 from the initial size of 24108. This is a great way to reduce over fitting. 

I finally randomly shuffled the data set and put 0.196% of the data into a validation set. Training set size : 8668.

While feeding the data to the model, I did further preprocessing of individual image by normalization and gray scaling in keras itself. By this way, excessive/unhelpful information was removed from each image data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by final validation loss of 0.0086. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

> Best regards, 
> Vivek Mano
