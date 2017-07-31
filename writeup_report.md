# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_resources/model.png "Model Visualization"
[image2]: ./writeup_resources/example1.jpg "Grayscaling"
[image3]: ./writeup_resources/example2_1.jpg "Recovery Image"
[image4]: ./writeup_resources/example2_2.jpg "Recovery Image"
[image5]: ./writeup_resources/example2_3.jpg "Recovery Image"
[image6]: ./writeup_resources/example3_1.jpg "Normal Image"
[image7]: ./writeup_resources/example3_2.jpg "Flipped Image"
[image8]: ./writeup_resources/example4_1.jpg "Flipped Image"
[image9]: ./writeup_resources/example4_2.jpg "Cropped Image"
[image10]: ./writeup_resources/example5_1.jpg "Left Camera"
[image11]: ./writeup_resources/example5_2.jpg "Center Camera"
[image12]: ./writeup_resources/example5_3.jpg "Right Camera"

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results
* `video.mp4` demonstrating the model in action on the first track of the simulator.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model.py is used to train the model using images captured from the simulator by executing
```sh
python model.py --drive_data
```
More advanced options can be used, and those can be viewed by issuing
```sh
python model.py --help
```
Which will present the following:
```
usage: model.py [-h] [--drive_data DRIVE_DATA] [--out OUT]
                [--learning_rate LEARNING_RATE] [--dropout_rate DROPOUT_RATE]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--load_model [LOAD_MODEL]] [--noload_model] [--trace [TRACE]]
                [--notrace]

optional arguments:
  -h, --help            show this help message and exit
  --drive_data DRIVE_DATA
                        Path to folder which contains drive data from the
                        simulator
  --out OUT             Output path from model file.
  --learning_rate LEARNING_RATE
                        The learning rate of the model.
  --dropout_rate DROPOUT_RATE
                        The dropout rate of the model.
  --epochs EPOCHS       The number of epochs used for training the model
  --batch_size BATCH_SIZE
                        The number of samples in each batch
  --load_model [LOAD_MODEL]
                        If `True` load model from last runs.
  --noload_model
  --trace [TRACE]       If `True` display extended tracebacks on exception.
  --notrace
```
#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a preprocessing region which contains a cropping layer, image resizing layer and a normalization layer.
Those layers are used to make the image fit the original NVIDA model and normalize the image data before entering into the convolution layers. (model.py lines 57-61)

Later, the image data passes through three 5x5 convolution layers
with a 2x2 stride, followed by a 3x3 convolution, a dropout layer and anther 3x3 convolution. Both 3x3 convolution layers have 1x1 stride.
(model.py lines 63-73)

And finally the convoluted data is flattened and passes through four fully connected layers which the last of them is giving the steering data prediction. (model.py lines 75-83)


The model includes RELU activations to introduce nonlinearity between the layers.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 71).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 265-274). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (using `drive.py`)

#### 3. Model parameter tuning

The model used an adam optimizer with default learning rate of 0.001.
The learning rate can be tuned manually by the *--learning_rate* argument (model.py line 258).

The batch size, number of epochs and dropout rate can also be tuned using the script arguments.

When training the model, I used the default values for all parameters:
* learning rate: 0.001
* epochs: 10
* batch size: 128
* dropout rate: 0.6

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track in the opposite direction and including data of the second track.

All data samples from the simulator contained three images from different
angles in order to provide more data for the turns. The images from the
left and right cameras was added to the data set with bias in order to
fit the steering data from the center camera.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to used
a model that worked for others for a similar purpose task.
That's why I used NVIDIA DAVE 2 model from NVIDIA "End to End Learning for Self-Driving Cars" [article](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


My first step was to use a convolution layers because they are a good fit for image data. They can generalize the data and make it easier for the fully connected layers that follow to make predictions based on that data.

After I built the model I added layers for preprocessing in order to fit
the NVIDIA model as best as I can.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In order to check for overfitting I split the data to training and validation again at the start of each epoch. This way I can eliminate a scenario in which a certain split validation set is overfitting due to high correlation with the training set data.

To combat the overfitting even more, I modified the model and added a dropout layer in the middle of it.

Then I trained the model for a couple of times on the same data, by calling the model.py script multiple times and testing the model performance on the simulator between calls.

At the beginning there were a few spots where the vehicle fell off the track, I tried at first to turn the images to grayscale to improve the driving behavior in these cases but as I found out it only made things worse, I think because it was hard for the model to differentiate between the asphalt and the sand.

In the end I reverted the grayscale conversion and called `model.py` 2-3 times, this process trained the model enough to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

In order to help the model generalize better I recorded one lap on track one in the opposite direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover from the sides of the track.
These images show what a recovery looks like starting from the right side of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded one lap on track two in order to improve the model generalize even better.

To augment the data sat, I also flipped images and angles thinking that this would create data for both steering in both directions.
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

In addition, the simulator provides us with images from three cameras: left, center and right. Which can be used to augment more data by shifting the steering angle for the left and right cameras.

![alt text][image10]
![alt text][image11]
![alt text][image12]


After the collection process, I had X number of data points. I then preprocessed this data by cropping, resizing and normalizing the pixels.
Example for cropping:
![alt text][image8]
![alt text][image9]

When loading the images I dropped 70% of the data with the car going straight, otherwise the model will have bias for going straight because most of the collected data is with little to no turning.

Finally, for each epoch I randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
