# Coins Detector
## Using Watson's Visual Recognition + OpenCV in Python 3.X

A Euro detector which use a custom classifier trained with the zips in the train folder.

For a deeper a complex description go to the [recipe](https://developer.ibm.com/recipes/tutorials/creating-a-coin-recognizer-with-watsons-visual-recognition-and-opencv-in-python3/), but basically you pass a picture to the program, it finds out if there's euros, in that case, then find every circle in the picture and pass it to the classifier again to find out its value.
