import numpy as np
import cv2
from scipy.stats import multivariate_normal
def norm_kernel(size = 9):
  # Define the mean and covariance matrix
  mean = [0, 0]
  cov = [[1, 0], [0, 1]]

  # Create a 2D normal distribution
  rv = multivariate_normal(mean, cov)

  # Generate a 2D array with values following the distribution
  x = np.linspace(-3, 3, size)
  y = np.linspace(-3, 3, size)
  X, Y = np.meshgrid(x, y)
  Z = rv.pdf(np.dstack((X, Y)))
  filter = Z - np.mean(Z)
  # Print the 2D array
  return filter

def amplify(label, strength = 3):
  label = label + strength * cv2.filter2D(label, -1, norm_kernel(5)) + strength * cv2.filter2D(label, -1, norm_kernel(9)) #cv2.erode(image, np.ones(9,9),iterations = 1 )
  return label
   

  
  