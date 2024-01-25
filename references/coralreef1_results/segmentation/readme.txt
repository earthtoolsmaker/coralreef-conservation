To run the notebook, you need to obtain the trained weights of the GoogLeNet model from the image classification
team and put them on your Google Drive, then puth the path in the model_weights_path variable.

Inference is done in two functions:

- get_coverage - takes one image and a classification model, return a segmented image and area percentage
covered by each class
- coverage_of_folder - takes all .jpg images in a given folder, calculates the coverage of each of them, returns
the average cover of each class. Be careful: when you run it on Colab with too many files, it crashes due to RAM limit.

The algorithm is entirely dependent on the classification model, performance can be improved by retraining
that model.

You might want to modify the SLIC parameters in the get_coverage function:
- n_segments: approximately how many clusters should the image be divided into
- sigma: how strong Gaussian noise should be applied before clustering: makes the cluster shapes more smooth and less detailed
- compactness: the higher the value, the more square-shaped and grid-like the clusters will be
