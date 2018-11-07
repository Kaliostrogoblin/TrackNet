import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):
    def _focal_loss(y_true, y_pred):
        labels = y_true
        logits = y_pred
        
        alpha_factor = tf.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)

        focal_weight = tf.where(tf.equal(labels, 1), 1 - logits, logits)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = tf.keras.backend.binary_crossentropy(labels, logits)
        cls_loss = focal_weight * cls_loss
        return cls_loss
    return _focal_loss


def ellipse_loss(y_true, y_pred):
    # x part
    x = (y_pred[:, 0] - y_true[:, 0]) / y_pred[:, 2]
    x = tf.square(x)
    # y part
    y = (y_pred[:, 1] - y_true[:, 1]) / y_pred[:, 3]
    y = tf.square(y)
    return tf.sqrt(x + y)


def radius_penalize(r1, r2):
    return r1 * r2


def custom_loss(lambda1 = 0.5, 
                lambda2 = 0.35, 
                lambda3 = 0.15,
                focal_alpha = 0.5, 
                focal_gamma = 1.0):
    # latent function
    def _custom_loss(y_true, y_pred):
        _focal_loss = focal_loss(focal_alpha, focal_gamma)
        # recalculate lambda1 
        lambda1_new = tf.maximum(lambda1, 1 - y_true[:, 0])
        # three parts of loss
        #entropy_part = lambda1 * K.binary_crossentropy(y_true[:, 0], y_pred[:, 0])
        entropy_part = lambda1_new * _focal_loss(y_true[:, 0], y_pred[:, 0])
        ellipse_part = lambda2 * ellipse_loss(y_true[:, 1:], y_pred[:, 1:])
        radius_part = lambda3 * radius_penalize(y_pred[:, 3], y_pred[:, 4])
        # summarize
        loss_value = entropy_part + y_true[:, 0] * (ellipse_part + radius_part)
        loss_value = tf.keras.backend.mean(loss_value, axis=0)
        return loss_value
    
    return _custom_loss
