
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(img)

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    return gray
    
    
    
    
    


def gen_sift_features(gray_img, eps=1e-7):
    sift = cv2.xfeatures2d.SIFT_create()
    # kps is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)

    # if there are no keypoints or descriptors, return an empty tuple
    if len(kp) == 0:
        return ([], None)
    
    
    # apply the Hellinger kernel by first L1-normalizing and taking the
    # square-root
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)
    #desc /= (np.linalg.norm(desc, axis=1, ord=2) + eps)
    
    return (kp, desc)

def show_sift_features(gray_img, color_img, kp):
    fig, ax = plt.subplots(figsize=(40, 20))
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


img = plt.imread('/Users/neo/wellth-wrk/desert_oasis/images/5053jCoi2AbTKzwonimqd4fWGxwqPyQS.jpg')
lesion_img_gray = to_gray(img)
plt.imshow(lesion_img_gray, cmap='gray');

# generate SIFT keypoints and descriptors
lesion_img_kp, lesion_img_desc = gen_sift_features(lesion_img_gray)


show_sift_features(lesion_img_gray, img, lesion_img_kp);
