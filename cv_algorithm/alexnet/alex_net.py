# coding:gbk
import tensorflow as tf

"""
网络结构：有5个卷积层，3个全连接层，且最后一个连接层有softmax层来分类
conv1卷积核11*11，3个channel，输出64维，stride为4，使用relu激活函数及最大池化，没有添加lrn层
conv2卷积核5*5,64个channel，输出192维，stride为1，使用relu激活函数及最大池化，没有添加lrn层
conv3卷积核3*3,192个channel，输出384维，stride为1，使用relu激活函数及最大池化
conv4卷积核3*3,384个channel，输出256维，stride为1，使用relu激活函数及最大池化
conv5卷积核3*3,256个channel，输出256维，stride为1，使用relu激活函数及最大池化
"""
batch_size = 1
image_size = 28
image = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], stddev=0.1, dtype=tf.float32))

def alex_net(image):
    with tf.name_scope('conv1') as scope:
        ## 第一层
        ## conv layer
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')
        lrn1 = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('conv2') as scope:
        ## 第二层
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([192], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')

        lrn2 = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3') as scope:
        ## 第三层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([384], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        pool3 = tf.nn.bias_add(conv, bias)

    with tf.name_scope('conv4') as scope:
        ## 第四层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1, dtype=tf.float32), trainable=True,
                           name='bias')
        pool4 = tf.nn.bias_add(conv, bias)

    with tf.name_scope('conv5') as scope:
        ## 第五层
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1, dtype=tf.float32), trainable=True,
                           name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')

        pool5 = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第六层全连接层
    pool5 = tf.reshape(pool5, (-1, 6 * 6 * 256))
    weight6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5, weight6), ful_bias1))

    # 第七层第二层全连接层
    weight7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1, weight7), ful_bias2))
    #
    # 第八层第三层全连接层
    weight8 = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1, dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]), name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2, weight8), ful_bias3))

    # softmax层
    weight9 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1), dtype=tf.float32, name="weight9")
    bias9 = tf.Variable(tf.constant(0.0, shape=[10]), dtype=tf.float32, name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3, weight9) + bias9)

if __name__ == '__main__':
    with tf.Session() as sess:
        init = tf.global_variables_initializer
        sess.run(init)
