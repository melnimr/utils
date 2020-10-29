import tensorflow as tf
import keras.backend as K

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
    
    # Test the metric
    
    import numpy as np
from sklearn.metrics import f1_score

# Samples
y_true = np.array([[1,1,0,0,1], [1,0,1,1,0], [0,1,1,0,0]])
y_pred = np.array([[0,1,1,1,1], [1,0,0,1,1], [1,0,1,0,0]])

print('Shape y_true:', y_true.shape)
print('Shape y_pred:', y_pred.shape)

# Results
print('sklearn Macro-F1-Score:', f1_score(y_true, y_pred, average='macro'))
print('Custom Macro-F1-Score:', K.eval(f1(y_true, y_pred)))
