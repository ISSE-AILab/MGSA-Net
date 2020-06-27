import tensorflow as tf
import cv2
import os
import h5py
import numpy as np
import tqdm

from weight_share_block import down_share,max_pool_comb,upscale_share,drop_comb
from layer_factory import conv_bn_relu_block,max_pooling_block,up_sample_concat_block


def trident_unet(input,input_128,bounding_box,keep_prob=0.5,d1=1,d2=2,d3=3,is_training=True,is_upscale=True,is_bn=True):
    dilation_rate=[d1,d2,d3]
    init_block=[input,input,input]

    # conv1_out=down_share(init_block,n_channl=64,kernel_size=[2,2],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv1',conv_name='conv_1')
    # conv2_out=down_share(conv1_out,n_channl=64,kernel_size=[2,2],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv2',conv_name='conv_2')
    # pool1_out=max_pool_comb(conv2_out)

    # conv3_out=down_share(pool1_out,n_channl=128,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv3',conv_name='conv_3')
    # conv4_out=down_share(conv3_out,n_channl=128,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv4',conv_name='conv_4')
    # pool2_out=max_pool_comb(conv4_out)

    # conv5_out=down_share(pool2_out,n_channl=256,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv5',conv_name='conv_5')
    # conv6_out=down_share(conv5_out,n_channl=256,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv6',conv_name='conv_6')
    # pool3_out=max_pool_comb(conv6_out)

    # conv7_out=down_share(pool3_out,n_channl=512,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv7',conv_name='conv_7')
    # conv8_out=down_share(conv7_out,n_channl=512,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv8',conv_name='conv_8')
    # pool4_out=max_pool_comb(conv8_out)

    # conv9_out=down_share(pool4_out,n_channl=1024,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv9',conv_name='conv_9')
    # conv10_out=down_share(conv9_out,n_channl=1024,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv10',conv_name='conv_10')
    # pool5_out=max_pool_comb(conv10_out)

    conv1_out=conv_bn_relu_block(input,n_channl=64,kernel_size=[2,2],is_training=is_training,is_bn=is_bn,name='conv_1')
    conv2_out=conv_bn_relu_block(conv1_out,n_channl=64,kernel_size=[2,2],is_training=is_training,is_bn=is_bn,name='conv_2')
    pool1_out=max_pooling_block(conv2_out)

    conv3_out=conv_bn_relu_block(pool1_out,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_3')
    conv4_out=conv_bn_relu_block(conv3_out,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_4')
    pool2_out=max_pooling_block(conv4_out)

    conv5_out=conv_bn_relu_block(pool2_out,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_5')
    conv6_out=conv_bn_relu_block(conv5_out,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_6')
    pool3_out=max_pooling_block(conv6_out)

    conv7_out=conv_bn_relu_block(pool3_out,n_channl=512,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_7')
    conv8_out=conv_bn_relu_block(conv7_out,n_channl=512,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_8')
    pool4_out=max_pooling_block(conv8_out)

    conv9_out=conv_bn_relu_block(pool4_out,n_channl=1024,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_9')
    conv10_out=conv_bn_relu_block(conv9_out,n_channl=1024,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='conv_10')

    up1=upscale_share([conv10_out,conv8_out],[conv10_out,conv8_out],[conv10_out,conv8_out],
                    n_channl=512,kernel_size=[2,2],dilation_rate=dilation_rate,is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,share_name='deconv1',conv_name='deconv_1')
    conv11_out=down_share(up1,n_channl=256,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv11',conv_name='conv_11')
    conv12_out=down_share(conv11_out,n_channl=256,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv12',conv_name='conv_12')

    up2=upscale_share([conv12_out[0],conv6_out],[conv12_out[1],conv6_out],[conv12_out[2],conv6_out],
                    n_channl=256,kernel_size=[2,2],dilation_rate=dilation_rate,is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,share_name='deconv2',conv_name='deconv_2')
    conv13_out=down_share(up2,n_channl=128,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv13',conv_name='conv_13')
    conv14_out=down_share(conv13_out,n_channl=128,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv14',conv_name='conv_14')

    up3=upscale_share([conv14_out[0],conv4_out],[conv14_out[1],conv4_out],[conv14_out[2],conv4_out],
                    n_channl=128,kernel_size=[2,2],dilation_rate=dilation_rate,is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,share_name='deconv3',conv_name='deconv_3')
    conv15_out=down_share(up3,n_channl=64,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv15',conv_name='conv_15')
    conv16_out=down_share(conv15_out,n_channl=64,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv16',conv_name='conv_16')

    up4=upscale_share([conv16_out[0],conv2_out],[conv16_out[1],conv2_out],[conv16_out[2],conv2_out],
                    n_channl=64,kernel_size=[2,2],dilation_rate=dilation_rate,is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,share_name='deconv4',conv_name='deconv_4')
    conv17_out=down_share(up4,n_channl=32,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv17',conv_name='conv_17')
    conv18_out=down_share(conv17_out,n_channl=32,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv18',conv_name='conv_18')

    conv19_out=down_share(conv18_out,n_channl=2,kernel_size=[3,3],branch_dilation=dilation_rate,is_training=is_training,is_bn=is_bn,share_name='conv19',conv_name='conv_19')
    conv20_out=down_share(conv19_out,n_channl=1,kernel_size=[1,1],branch_dilation=dilation_rate,activation='sigmoid',is_training=is_training,is_bn=is_bn,share_name='conv20',conv_name='conv_20')
    
    # 128 input 
    convU1=conv_bn_relu_block(input_128,n_channl=64,kernel_size=[2,2],is_training=is_training,is_bn=is_bn,name='cu1')
    convU2=conv_bn_relu_block(convU1,n_channl=64,kernel_size=[2,2],is_training=is_training,is_bn=is_bn,name='cu2')
    p1=max_pooling_block(convU2)

    convU3=conv_bn_relu_block(p1,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu3')
    convU4=conv_bn_relu_block(convU3,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu4')
    p2=max_pooling_block(convU4)

    convU5=conv_bn_relu_block(p2,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu5')
    convU6=conv_bn_relu_block(convU5,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu6')
    p3=max_pooling_block(convU6)

    convU7=conv_bn_relu_block(p3,n_channl=512,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu7')
    convU8=conv_bn_relu_block(convU7,n_channl=512,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu8')
    p4=max_pooling_block(convU8)

    convU9=conv_bn_relu_block(p4,n_channl=1024,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu9')
    convU10=conv_bn_relu_block(convU9,n_channl=1024,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu10')

    up1=up_sample_concat_block(convU10,convU8,n_channl=512,kernel_size=[2,2],is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,name='du1')
    convU11=conv_bn_relu_block(up1,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu11')
    convU12=conv_bn_relu_block(convU11,n_channl=256,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu12')

    up2=up_sample_concat_block(convU12,convU6,n_channl=256,kernel_size=[2,2],is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,name='du2')
    convU13=conv_bn_relu_block(up2,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu13')
    convU14=conv_bn_relu_block(convU13,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu14')

    up3=up_sample_concat_block(convU14,convU4,n_channl=128,kernel_size=[2,2],is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,name='du3')
    convU15=conv_bn_relu_block(up3,n_channl=64,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu15')
    convU16=conv_bn_relu_block(convU15,n_channl=64,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu16')

    up4=up_sample_concat_block(convU16,convU2,n_channl=64,kernel_size=[2,2],is_training=is_training,is_upscale=is_upscale,is_bn=is_bn,name='du4')
    box_ids = tf.range(0, tf.shape(bounding_box)[0])
    # crop_img1 = tf.image.crop_and_resize(conv18_out[0], bounding_box, box_ids,crop_size=(128,128),name="crop_image")
    crop_img = tf.image.crop_and_resize(conv18_out[1], bounding_box, box_ids,crop_size=(96,96),name="crop_image")
    # crop_img3 = tf.image.crop_and_resize(conv18_out[2], bounding_box, box_ids,crop_size=(128,128),name="crop_image")
    conv_comb=tf.concat([up4,crop_img],axis=-1)

    convU17=conv_bn_relu_block(conv_comb,n_channl=32,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu17')
    convU18=conv_bn_relu_block(convU17,n_channl=32,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu18')

    # conv_comb=tf.concat([conv_comb,crop_img2],axis=-1)
    # conv_comb=tf.concat([conv_comb,crop_img3],axis=-1)

    convU19=conv_bn_relu_block(convU18,n_channl=2,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='cu19')
    convU20=conv_bn_relu_block(convU19,n_channl=1,kernel_size=[1,1],activation='sigmoid',is_training=is_training,is_bn=is_bn,name='cu20')

    return conv20_out,convU20

# x=tf.placeholder(tf.float32,[None,256,256,1])
# x_128=tf.placeholder(tf.float32,[None,96,96,1])
# bbox=tf.placeholder(tf.float32,[None,4])
# y=trident_unet(x,x_128,bbox)
# print(y)

def _next_batch(train_images_f, train_labels_f,train_128_images_f,train_128_labels_f,bounding_box_f,batch_size,index_in_epoch):
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size
    num_examples = train_images_f.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images_f = train_images_f[perm]
        train_labels_f = train_labels_f[perm]
        train_128_images_f=train_128_images_f[perm]
        train_128_labels_f=train_128_labels_f[perm]
        bounding_box_f=bounding_box_f[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch

    train_images=train_images_f[start:end]
    train_labels=train_labels_f[start:end]
    train_128_images=train_128_images_f[start:end]
    train_128_labels=train_128_labels_f[start:end]
    bounding_box=bounding_box_f[start:end]

    bx=[]
    by=[]
    for i in range(len(train_images)):
        image = cv2.imread(''.join(train_images[i]), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(''.join(train_labels[i]), cv2.IMREAD_GRAYSCALE)

        # size = (int(image.shape[0]*0.5), int(image.shape[1]*0.5))  
        # image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # label = cv2.resize(label, size, interpolation=cv2.INTER_AREA)

        bx.append(image)
        by.append(label)
    bx=np.array(bx)[:,:,:,np.newaxis]
    by=np.array(by)[:,:,:,np.newaxis]

    bx_128=[]
    by_128=[]
    for i in range(len(train_128_images)):
        image = cv2.imread(''.join(train_128_images[i]), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(''.join(train_128_labels[i]), cv2.IMREAD_GRAYSCALE)
        bx_128.append(image)
        by_128.append(label)
    bx_128=np.array(bx_128)[:,:,:,np.newaxis]
    by_128=np.array(by_128)[:,:,:,np.newaxis]

    bbox=[]
    for i in range(len(bounding_box)):
        temp=[bounding_box[i][1],bounding_box[i][2],bounding_box[i][3],bounding_box[i][4]]
        bbox.append(temp)
    bbox=np.array(bbox)

    return train_images_f, train_labels_f,train_128_images_f,train_128_labels_f,bounding_box_f,bx,by,bx_128,by_128,bbox,index_in_epoch


class trident_Model(object):
    def __init__(self, image_height, image_width,image_128_height,image_128_width, channels=1, costname=("dice coefficient",),
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_128_height=image_128_height
        self.image_128_width=image_128_width
        self.channels = channels

        self.X = tf.placeholder("float", shape=[None, self.image_height, self.image_width,
                                                self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.image_height, self.image_width,
                                                   self.channels])

        self.X_128 = tf.placeholder("float", shape=[None, self.image_128_height, self.image_128_width,
                                                self.channels])
        self.Y_128_gt = tf.placeholder("float", shape=[None, self.image_128_height, self.image_128_width,
                                                   self.channels])
        self.bounding_box= tf.placeholder("float",shape=[None,4])
        self.lr = tf.placeholder('float')
        self.keep_prob=tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)

        self.Y_pred_list,self.Y_128_pred= trident_unet(self.X,self.X_128,self.bounding_box,self.keep_prob)
        self.cost = self.__get_cost(costname[0])
        self.accuracy = -self.__get_cost(costname[0])

        self.Y_test,self.X_test,self.X_128_test,self.Y_128_test,self.bbox_test=self.__get_pred_img()

        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_pred_img(self):
        src_path = "/home/zhangxiangbo/trident_unet_comb/data/img_256/LIDC-IDRI-0337_1_3124.png"
        mask_path = "/home/zhangxiangbo/trident_unet_comb/data/mask_256/LIDC-IDRI-0337_1_3124_mask.png"
        src_128="/home/zhangxiangbo/trident_unet_comb/data/img_96/LIDC-IDRI-0337_1_3124.png"
        mask_128="/home/zhangxiangbo/trident_unet_comb/data/mask_96/LIDC-IDRI-0337_1_3124_mask.png"
        
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # size = (int(img.shape[0]*0.5), int(img.shape[1]*0.5))  
        # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)

        img_128 = cv2.imread(src_128, cv2.IMREAD_GRAYSCALE)
        mask_128 = cv2.imread(mask_128, cv2.IMREAD_GRAYSCALE)

        test_imges=np.array(img)[np.newaxis,:,:,np.newaxis]/255.
        test_masks=np.array(mask)[np.newaxis,:,:,np.newaxis]/255.
        test_128_imges=np.array(img_128)[np.newaxis,:,:,np.newaxis]/255.
        test_128_masks=np.array(mask_128)[np.newaxis,:,:,np.newaxis]/255.

        bbox_test=[[67,0,163,96]]
        bbox_test=np.array(bbox_test)
        bbox_test=np.multiply(bbox_test,1.0/256.0)

        return test_masks,test_imges,test_128_imges,test_128_masks,bbox_test
 
    def __get_cost(self, cost_name):
        H, W, C = self.X.get_shape().as_list()[1:]
        H_s, W_s, C_s = self.X_128.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            
            pred_flat_d1 = tf.reshape(self.Y_pred_list[0], [-1, H * W * C ])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C ])
            intersection_d1 = 2 * tf.reduce_sum(pred_flat_d1 * true_flat, axis=1) + smooth
            denominator_d1 = tf.reduce_sum(pred_flat_d1, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss_d1 = 1-tf.reduce_mean(intersection_d1 / denominator_d1)

            pred_flat_d2 = tf.reshape(self.Y_pred_list[1], [-1, H * W * C ])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C ])
            intersection_d2 = 2 * tf.reduce_sum(pred_flat_d2 * true_flat, axis=1) + smooth
            denominator_d2 = tf.reduce_sum(pred_flat_d2, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss_d2 = 1-tf.reduce_mean(intersection_d2 / denominator_d2)

            pred_flat_d3 = tf.reshape(self.Y_pred_list[2], [-1, H * W * C ])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C ])
            intersection_d3 = 2 * tf.reduce_sum(pred_flat_d3 * true_flat, axis=1) + smooth
            denominator_d3 = tf.reduce_sum(pred_flat_d3, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss_d3 = 1-tf.reduce_mean(intersection_d3 / denominator_d3)

            pred_flat_df = tf.reshape(self.Y_128_pred, [-1, H_s* W_s * C_s ])
            true_flat_f = tf.reshape(self.Y_128_gt, [-1, H_s * W_s * C_s ])
            intersection_df = 2 * tf.reduce_sum(pred_flat_df * true_flat_f, axis=1) + smooth
            denominator_df = tf.reduce_sum(pred_flat_df, axis=1) + tf.reduce_sum(true_flat_f, axis=1) + smooth
            loss_df = 1-tf.reduce_mean(intersection_df / denominator_df)

            # eps=1e-5
            # predict=self.Y_predf
            # target=self.Y_gt
            # intersection = tf.reduce_sum(predict * target, axis=1)
            # intersection = tf.reduce_sum(intersection,axis=1)
            # union = tf.reduce_sum(predict * predict + target * target, axis=1)
            # union = tf.reduce_sum(union,axis=1)
            # dice = (2. * intersection + eps) / (union + eps)
            # opt=tf.reduce_mean(dice, axis=0)
            # loss=tf.reduce_mean(1-opt)

            loss=loss_df+loss_d1+loss_d2+loss_d3
            # loss=1-loss_df

        return loss

    def train(self, train_images, train_labels,train_128_images,train_128_labels,bounding_box, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model/"):
            os.makedirs(logs_path + "model/")
        if not os.path.exists(logs_path + "test_img/"):
            os.makedirs(logs_path + "test_img/")
        model_path = logs_path + "model/" + model_path

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)
        print("session init finished!")

        DISPLAY_STEP = 1
        index_in_epoch = 0
        score=4
        epoch=0


        train_epochs = train_images.shape[0] * train_epochs
        print("********************",train_epochs)
        for i in range(train_epochs):
            # get new batch
            train_images,train_labels,train_128_images,train_128_labels,bounding_box,batch_xs, batch_ys,batch_128_xs,batch_128_ys,bbox,index_in_epoch=_next_batch(train_images,
                                                                                                                                                                    train_labels,
                                                                                                                                                                    train_128_images,
                                                                                                                                                                    train_128_labels,
                                                                                                                                                                    bounding_box,
                                                                                                                                                                    batch_size,
                                                                                                                                                                    index_in_epoch)
            # assert batch_xs.shape==(batch_size,512,512,1)
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            batch_128_xs = batch_128_xs.astype(np.float)
            batch_128_ys = batch_128_ys.astype(np.float)
            bbox = bbox.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            batch_128_xs = np.multiply(batch_128_xs, 1.0 / 255.0)
            batch_128_ys = np.multiply(batch_128_ys, 1.0 / 255.0)
            bbox = np.multiply(bbox, 1.0 / 256.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step

            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run([self.cost, self.accuracy],
                                                      feed_dict={self.X: batch_xs,
                                                                self.Y_gt: batch_ys,
                                                                self.X_128:batch_128_xs,
                                                                self.Y_128_gt:batch_128_ys,
                                                                self.bounding_box:bbox,
                                                                self.lr: learning_rate,
                                                                self.phase: 1,
                                                                self.keep_prob: dropout_conv})#self.lr: learning_rate,
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))

                pred_list,pred,cost= sess.run([self.Y_pred_list,self.Y_128_pred,self.cost], feed_dict={self.X: self.X_test,
                                                                                                        self.Y_gt: self.Y_test,
                                                                                                        self.X_128:self.X_128_test,
                                                                                                        self.Y_128_gt:self.Y_128_test,
                                                                                                        self.bounding_box:self.bbox_test,
                                                                                                        self.phase: 1,
                                                                                                        self.lr: learning_rate,
                                                                                                        self.keep_prob: 1})
                print('loss for test:', cost)
                result=np.array(pred_list).squeeze()
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite(logs_path + 'test_img/'+'test_'+str(i)+'_0.png',result[0])
                cv2.imwrite(logs_path + 'test_img/'+'test_'+str(i)+'_1.png',result[1])
                cv2.imwrite(logs_path + 'test_img/'+'test_'+str(i)+'_2.png',result[2])

                result=np.array(pred).squeeze()
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite(logs_path + 'test_img/'+'test_'+str(i)+'_p.png',result)

                # result=np.array(trilist).squeeze()
                # result = result.astype(np.float32) * 255.
                # result = np.clip(result, 0, 255).astype('uint8')
                # cv2.imwrite('./log/test_img/test_'+str(i)+'_bran1.bmp',result[0])
                # cv2.imwrite('./log/test_img/test_'+str(i)+'_bran2.bmp',result[1])
                # cv2.imwrite('./log/test_img/test_'+str(i)+'_bran3.bmp',result[2])
                

                # save_path = saver.save(sess, model_path, global_step=i)
                # print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.X_128:batch_128_xs,
                                                                            self.Y_128_gt:batch_128_ys,
                                                                            self.bounding_box:bbox,
                                                                            # self.global_:i,
                                                                            self.lr: learning_rate,
                                                                            self.keep_prob: dropout_conv})
            summary_writer.add_summary(summary, i)
            if (i+1)%(train_images.shape[0]//batch_size)==0:
                print("testing sorce...")
                cost=test_score(self,sess)
                print("average cost :",cost)
                epoch+=1
                with open("./record_changePath_resize.csv","a") as f:
                    f.write(str(epoch)+","+str(cost)+","+str(i)+"\n")
                if(cost<score):
                    score=cost
                    save_path = saver.save(sess, model_path, global_step=i)
                    print("Model saved in file:", save_path)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images,test_masks,test_128_images,test_128_masks,bbox):
        test_images = test_images.astype(np.float)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        test_masks = test_masks.astype(np.float)
        test_masks = np.multiply(test_masks, 1.0 / 255.0)

        test_128_images = test_128_images.astype(np.float)
        test_128_images = np.multiply(test_128_images, 1.0 / 255.0)

        test_128_masks = test_128_masks.astype(np.float)
        test_128_masks = np.multiply(test_128_masks, 1.0 / 255.0)

        bbox = bbox.astype(np.float)
        bbox = np.multiply(bbox, 1.0 / 256.0)

        pred_list,pred_128 = self.sess.run([self.Y_pred_list,self.Y_128_pred], feed_dict={self.X: test_images,
                                                                                                        self.Y_gt: test_masks,
                                                                                                        self.X_128:test_128_images,
                                                                                                        self.Y_128_gt:test_128_masks,
                                                                                                        self.bounding_box:bbox,
                                                                                                        self.phase: 1,   
                                                                                                        self.keep_prob: 1})
        pred_list=np.array(pred_list).squeeze()
        pred_list = pred_list.astype(np.float32) * 255.
        pred_list = np.clip(pred_list, 0, 255).astype('uint8')

        pred_128=np.array(pred_128).squeeze()
        pred_128 = pred_128.astype(np.float32) * 255.
        pred_128 = np.clip(pred_128, 0, 255).astype('uint8')

        return pred_list,pred_128

def test_score(self,sess):
    f=h5py.File("./256_96_test.h5","r")
    test_images=f["X_256"]
    test_masks=f["Y_256"]
    test_96_images=f["X_96"]
    test_96_masks=f["Y_96"]
    bbox=f["bbox"]

    test_images=np.array(test_images)
    test_images = test_images.astype(np.float)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    test_masks=np.array(test_masks)
    test_masks = test_masks.astype(np.float)
    test_masks = np.multiply(test_masks, 1.0 / 255.0)

    test_96_images=np.array(test_96_images)
    test_96_images = test_96_images.astype(np.float)
    test_96_images = np.multiply(test_96_images, 1.0 / 255.0)

    test_96_masks=np.array(test_96_masks)
    test_96_masks = test_96_masks.astype(np.float)
    test_96_masks = np.multiply(test_96_masks, 1.0 / 255.0)

    bbox=np.array(bbox)
    bbox=bbox.tolist()
    bbox = np.multiply(bbox, 1.0 / 256.0)
    cost=0
    for i in tqdm.tqdm(range(test_images.shape[0])):
        sub_cost = sess.run([self.cost], feed_dict={self.X: test_images[i][np.newaxis,:,:,:],
                                                                    self.Y_gt: test_masks[i][np.newaxis,:,:,:],
                                                                    self.X_128:test_96_images[i][np.newaxis,:,:,:],
                                                                    self.Y_128_gt:test_96_masks[i][np.newaxis,:,:,:],
                                                                    self.bounding_box:bbox[i],
                                                                    self.phase: 1,   
                                                                    self.keep_prob: 1})
        cost+=sub_cost[0]
    cost=cost/test_images.shape[0]
    return cost


