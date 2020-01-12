# focal loss with multi label
def focal_loss(classes_num, gamma=2.,  e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        #ones = array_ops.ones_like(prediction_tensor, dtype=prediction_tensor.dtype)
        #one_minus_p=array_ops.where(tf.greater(target_tensor,zeros), ones-target_tensor*prediction_tensor, ones)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))*target_tensor

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor
        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)
        balanced_fl = alpha * FT
        balanced_fl = 10*tf.reduce_mean(FT)

        fianal_loss = balanced_fl 
        return fianal_loss
    return focal_loss_fixed
    
import keras.backend as K  
def categorical_squared_hinge(y_true, y_pred):
    """
    hinge with 0.5*W^2 ,SVM
    """
    y_true = 2. * y_true - 1 # 
    vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss
#    vvv = K.square(vvvv) #
    vv = K.sum(vvvv, 1, keepdims=False) 
    v = K.mean(vv, axis=-1)
    return v