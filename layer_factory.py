import tensorflow as tf

def conv_bn_relu_block(input,
            n_channl=8,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            dilation_rate=1,
            activation=None,
            is_training=True,
            is_bn=True,
            name='conv'):
    if is_bn:
        padding=padding.upper()
        conv=tf.layers.conv2d(input,
                            n_channl,
                            kernel_size,
                            strides=strides,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding=padding,
                            dilation_rate=dilation_rate,
                            activation=None,
                            name=name)
        conv_norm=tf.layers.batch_normalization(conv, training=is_training)
        if activation==None:
            conv=tf.nn.relu(conv_norm)
            # conv=tf.nn.relu6(conv_norm)
        else:
            conv=tf.nn.sigmoid(conv_norm)
    else:
        padding=padding.upper()
        conv=tf.layers.conv2d(input,
                            n_channl,
                            kernel_size,
                            strides=strides,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding=padding,
                            dilation_rate=dilation_rate,
                            activation=None,
                            name=name)
        if activation==None:
            conv=tf.nn.relu(conv)
            # conv=tf.nn.relu6(conv)
        else:
            conv=tf.nn.sigmoid(conv)
    return conv

def max_pooling_block(input,pool_size=(2,2),strides=[2,2]):
    max_conv=tf.layers.max_pooling2d(input,pool_size=pool_size,strides=strides)
    return max_conv

def up_sample_concat_block(input,
                        concat_conv,
                        n_channl=8,
                        kernel_size=[2,2],
                        strides=[1,1],
                        padding='same',
                        dilation_rate=1,
                        activation=None,
                        is_training=True,
                        is_upscale=True,
                        is_bn=True,
                        name='deconv'):
    
    padding=padding.upper()
    if is_upscale :

        size_h = 2 * int(input.get_shape()[-2])
        size_w = 2 * int(input.get_shape()[-2])
        _size = [size_h, size_w]

        up_image=tf.image.resize_images(input,_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv=conv_bn_relu_block(
            input=up_image,
            n_channl=n_channl,
            kernel_size=kernel_size,
            padding=padding,
            dilation_rate=dilation_rate,
            is_training=is_training,
            name=name,
            is_bn=is_bn
        )
        cc_conv=tf.concat([concat_conv,conv],axis=-1)
        return cc_conv
    else:
        if is_bn:
            conv = tf.layers.conv2d_transpose(input,
                                            n_channl, 
                                            kernel_size=kernel_size,
                                            strides=[2,2],
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            activation=None,
                                            padding=padding, 
                                            name=name)
            conv_norm=tf.layers.batch_normalization(conv, training=is_training)
            conv=tf.nn.relu(conv_norm)
            # conv=tf.nn.relu6(conv_norm)
        else:
            conv = tf.layers.conv2d_transpose(input,
                                            n_channl, 
                                            kernel_size=kernel_size,
                                            strides=[2,2],
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            activation=None,
                                            padding=padding, 
                                            name=name)
            conv=tf.nn.relu(conv)
            # conv=tf.nn.relu6(conv)
        cc_conv=tf.concat([concat_conv,conv],axis=-1)
        return cc_conv
