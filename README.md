# Aruco-CV2

The code identifies aruco marker in a given image. (5*5) Markers
Once the aruco markers are identified the program can project either a cube or a cylinder or draw the axes for the aruco marker.

## TO RUN
* import the module.
* Call the funtion by passing an image mtrix (numpy).
* Can be connected to a video input and be used in real time.


## The draw functions

* The main input is a numpy array representing the image with the aruco markers. The code will identify the positin and orientation of the aruco markers and will overlay 3D objects on it.
* The other parameters are used to correct any distortions caused by the webcam. This parameter can be ignored.
* The function will return the modified image. The thickness and color of the objects drawn can be changed if necessary.
* This code was written before the invention of loops.
