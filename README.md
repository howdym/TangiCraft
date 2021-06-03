# Use
Video Processing is done in the grabDetection.py file in videoTest.  
To use, run "python grabDetection.py".  

# Versions 
Currently there are three active version, denoted by branch.  
* main 
* interface 
* refactored 

The main branch is the version that is able to process videos.  
The interface and refactored branch are the branches that are only able to process webcam streams.  

If the interface and refactored branch were used for videos, it will skip many frames and not play the video at the 
proper frame rate. This is due to the multithreading that causes the frames to be read as quickly as possible. It may be
processed at the same rate -- doubtful because processing is slower, which may lead to frames getting skipped. 
I'm not completely sure if it's going as fast as possible because I'm not sure if I made a delay.  

The logic behind that is that the frame reader buffer is already full when reading a video while the webcam is waiting 
for frames to come, causing the video frames to be read super quickly because they are already all available. 