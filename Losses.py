from keras import backend as K
from keras.backend import maximum

#print(list(word_index.keys())[list(word_index.values()).index(0)])

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        if(K.backend()=="tensorflow"):
            import tensorflow as tf
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        if(K.backend()=="theano"):
            import theano.tensor as T
            pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

def focal_Loss(target,output,gamma=2):
	output/=K.sum(output,axis=-1,keepdims=True)
	eps=K.epsilon()
	output=K.clip(output,eps,1.-eps)
	return -K.sum(K.pow(1.-output,gamma)*target*K.log(output),axis=-1)

def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    y_pred_sq = K.sqrt(y_pred)
    y_pred_sum = K.sum(y_pred_sq)
    y_pred_new = y_pred_sq/y_pred_sum
    return K.sum(y_true * K.log(y_true / y_pred_new), axis=-1)

def combined_kld_focal(y_true,y_pred,lambda_val_1,lambda_val_2):
	return maximum(focal_Loss(y_true,y_pred,3),kld(y_true,y_pred))


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """    
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) +
                   y_true * K.square(K.maximum(margin - y_pred, 0)))

ef my_cosine_proximity(y_true, y_pred):
    a = y_pred[0]
    b = y_pred[1]
    # depends on whether you want to normalize
    a = K.l2_normalize(a, axis=-1)
    b = K.l2_normalize(b, axis=-1)        
    return -K.mean(a * b, axis=-1) + 0 * y_true






