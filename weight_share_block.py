import tensorflow as tf
from layer_factory import conv_bn_relu_block,max_pooling_block,up_sample_concat_block
def down_share(input_list,
            n_channl=8,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            branch_dilation=[1,2,3],
            is_training=True,
            is_bn=True, 
            activation=None,
            share_name='conv',
            conv_name='conv_name'):
    
    with tf.variable_scope(share_name):
        branch1_out = conv_bn_relu_block(input_list[0],n_channl=n_channl,kernel_size=kernel_size,activation=activation,padding=padding,dilation_rate=branch_dilation[0],is_training=is_training,is_bn=is_bn,name=conv_name)
    with tf.variable_scope(share_name, reuse=True):
        branch2_out = conv_bn_relu_block(input_list[1],n_channl=n_channl,kernel_size=kernel_size,activation=activation,padding=padding,dilation_rate=branch_dilation[1],is_training=is_training,is_bn=is_bn,name=conv_name)
    with tf.variable_scope(share_name, reuse=True):
        branch3_out = conv_bn_relu_block(input_list[2],n_channl=n_channl,kernel_size=kernel_size,activation=activation,padding=padding,dilation_rate=branch_dilation[2],is_training=is_training,is_bn=is_bn,name=conv_name)

    return [branch1_out,branch2_out,branch3_out]

def max_pool_comb(input_list,
                pool_size=[2,2],
                strides=[2,2]):
    
    branch1_pool_out=max_pooling_block(input_list[0],pool_size=pool_size,strides=strides)
    branch2_pool_out=max_pooling_block(input_list[1],pool_size=pool_size,strides=strides)
    branch3_pool_out=max_pooling_block(input_list[2],pool_size=pool_size,strides=strides)

    return [branch1_pool_out,branch2_pool_out,branch3_pool_out]


def drop_comb(input_list,keep_prob=0.5):
    
    branch1_drop_out=tf.layer.drop(input_list[0],rate=keep_prob)
    branch2_drop_out=tf.layer.drop(input_list[1],rate=keep_prob)
    branch3_drop_out=tf.layer.drop(input_list[2],rate=keep_prob)

    return [branch1_drop_out,branch2_drop_out,branch3_drop_out]

def upscale_share(input1_list,
                input2_list,
                input3_list,
                n_channl=8,
                kernel_size=[2,2],
                strides=[1,1],
                padding='same',
                dilation_rate=[1,2,3],
                activation=None,
                is_training=True,
                is_upscale=True,
                is_bn=True,
                share_name='deconv',
                conv_name='deconv_name'):
    
    with tf.variable_scope(share_name):
        branch1_de_out = up_sample_concat_block(input1_list[0],input1_list[1],
                                        n_channl=n_channl,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilation_rate[0],
                                        activation=activation,
                                        is_training=is_training,
                                        is_upscale=is_upscale,
                                        is_bn=is_bn,
                                        name=conv_name)
    with tf.variable_scope(share_name, reuse=True):
        branch2_de_out = up_sample_concat_block(input2_list[0],input2_list[1],
                                        n_channl=n_channl,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilation_rate[1],
                                        activation=activation,
                                        is_training=is_training,
                                        is_upscale=is_upscale,
                                        is_bn=is_bn,
                                        name=conv_name)
    with tf.variable_scope(share_name, reuse=True):
        branch3_de_out = up_sample_concat_block(input3_list[0],input3_list[1],
                                        n_channl=n_channl,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        dilation_rate=dilation_rate[2],
                                        activation=activation,
                                        is_training=is_training,
                                        is_upscale=is_upscale,
                                        is_bn=is_bn,
                                        name=conv_name)
    
    return [branch1_de_out,branch2_de_out,branch3_de_out]


