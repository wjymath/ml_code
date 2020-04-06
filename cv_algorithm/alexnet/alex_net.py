# coding:gbk
import tensorflow as tf

"""
����ṹ����5������㣬3��ȫ���Ӳ㣬�����һ�����Ӳ���softmax��������
conv1�����11*11��3��channel�����64ά��strideΪ4��ʹ��relu����������ػ���û�����lrn��
conv2�����5*5,64��channel�����192ά��strideΪ1��ʹ��relu����������ػ���û�����lrn��
conv3�����3*3,192��channel�����384ά��strideΪ1��ʹ��relu����������ػ�
conv4�����3*3,384��channel�����256ά��strideΪ1��ʹ��relu����������ػ�
conv5�����3*3,256��channel�����256ά��strideΪ1��ʹ��relu����������ػ�
"""
batch_size = 1
image_size = 28
image = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], stddev=0.1, dtype=tf.float32))

def alex_net(image):
    with tf.name_scope('conv1') as scope:
        ## ��һ��
        ## conv layer
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')
        lrn1 = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('conv2') as scope:
        ## �ڶ���
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([192], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')

        lrn2 = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3') as scope:
        ## ������
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([384], mean=0, stddev=0.1, dtype=tf.float32), trainable=True, name='bias')
        pool3 = tf.nn.bias_add(conv, bias)

    with tf.name_scope('conv4') as scope:
        ## ���Ĳ�
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1, dtype=tf.float32), trainable=True,
                           name='bias')
        pool4 = tf.nn.bias_add(conv, bias)

    with tf.name_scope('conv5') as scope:
        ## �����
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0, stddev=0.1, dtype=tf.float32), name='filter')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1, dtype=tf.float32), trainable=True,
                           name='bias')
        net = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(net, name='relu')

        pool5 = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # ������ȫ���Ӳ�
    pool5 = tf.reshape(pool5, (-1, 6 * 6 * 256))
    weight6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5, weight6), ful_bias1))

    # ���߲�ڶ���ȫ���Ӳ�
    weight7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1, weight7), ful_bias2))
    #
    # �ڰ˲������ȫ���Ӳ�
    weight8 = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1, dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]), name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2, weight8), ful_bias3))

    # softmax��
    weight9 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1), dtype=tf.float32, name="weight9")
    bias9 = tf.Variable(tf.constant(0.0, shape=[10]), dtype=tf.float32, name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3, weight9) + bias9)

if __name__ == '__main__':
    with tf.Session() as sess:
        init = tf.global_variables_initializer
        sess.run(init)
