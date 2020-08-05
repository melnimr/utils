from pathlib import Path
import pandas as pd

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

import skimage.measure
import imageio
from PIL import Image
import requests
from io import BytesIO

from sklearn.model_selection import GroupKFold

from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

from glob import glob
import pandas as pd
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io

import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from glob import glob
import sklearn



import keras
import numpy as np
import tensorflow as tf
from keras.models import model_from_json, load_model
import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from functools import partial

import glob
import numpy as np
import cv2
from skimage import filters as skifilters
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
import tqdm
from sklearn.utils import shuffle
import pandas as pd

import os
import h5py
import time
import json
import warnings
from PIL import Image

from sklearn.metrics import accuracy_score, roc_auc_score
import pdb
import matplotlib.pyplot as plt

import pickle 
import os






image = cv2.imread('/Users/neo/wellth-wrk/desert_oasis/images/3DKYlxcLSOJYIn0NYUdmXWsDuZQA1NEE.jpg')

#image = img

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)





'''These contours are then sorted from left-to-right (allowing us to extract our reference object)'''
(cnts, _) = contours.sort_contours(cnts)
 # Remove contours which are not large enough
for k in range(0,20):
    try:
        cnts = [x for x in cnts if cv2.contourArea(x) > k]
        # Reference object dimensions
        # Here for reference I have used a 2cm x 2cm square
        mid = len(cnts)//2
        ref_object = cnts[mid]
    except:
        pass


orig = image.copy()
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")

box = perspective.order_points(box)

cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
