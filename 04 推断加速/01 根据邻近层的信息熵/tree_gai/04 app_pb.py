import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pprint
import time

def get_mnist():
    def load_data():
        mnist = input_data.read_data_sets(r'C:\Users\Cheng\.keras\datasets\t10k-images-idx3-ubyte', one_hot=True)
        return mnist

    mnist = load_data()
    test_set_x = mnist.test.images
    test_set_y = mnist.test.labels
    return test_set_x, test_set_y

def load_model(path):
    # 原始保存的pb文件
    graph_def = tf.GraphDef()
    with tf.Session() as sess:
        with tf.gfile.FastGFile(path,'wb+') as f:
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name='')
        sess.run(tf.global_variables_initializer())

        data = sess.graph.get_tensor_by_name("Placeholder:0")
        label = sess.graph.get_tensor_by_name("Placeholder_1:0")
        ratio = sess.graph.get_tensor_by_name("Placeholder_2:0")
        acc = sess.graph.get_tensor_by_name('accuracy:0')

        test_set_x, test_set_y = get_mnist()
        batch_size=1
        n_test_batches = test_set_x.shape[0]
        n_test_batches = int(n_test_batches / batch_size)

        valid_acc = 0
        costime=[]
        for i in range(n_test_batches):
            a=time.time()
            valid_acc = valid_acc + sess.run(acc, feed_dict={data: test_set_x[i * batch_size:(i + 1) * batch_size],
                                                            label: test_set_y[i * batch_size:(i + 1) * batch_size],
                                                            ratio: 1})
            costime.append(time.time()-a)

        test_acc = valid_acc / n_test_batches
        print('acc',test_acc)
        print('costime:',sum(costime[1:])/(n_test_batches-1))


def reload_model(path):
    graph_def = tf.GraphDef()
    with tf.Session() as sess:
        with tf.gfile.FastGFile(path,'wb+') as f:
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def,name='')
        sess.run(tf.global_variables_initializer())

        data = sess.graph.get_tensor_by_name("Placeholder:0")
        y = tf.placeholder(tf.float32, shape=[None, 10])
        features = sess.graph.get_tensor_by_name('BiasAdd:0')

        correct_prediction = tf.equal(tf.argmax(features, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        sts=20
        np.random.seed(2020)
        num_lst = np.random.randint(1,1000,sts,dtype=int)
        test_set_x, test_set_y = get_mnist()
        costime = []
        for nl in num_lst:
            a = time.time()
            acc =sess.run(accuracy,feed_dict={data:np.expand_dims(test_set_x[nl],axis=0),
                                              y:np.expand_dims(test_set_y[nl],axis=0)})
            pre_y = np.argmax(sess.run(features,feed_dict={data:np.expand_dims(test_set_x[nl],axis=0)}))
            print(np.expand_dims(test_set_y[nl],axis=0))
            costime.append(time.time()-a)
            tr_y = np.argmax(test_set_y[nl])
            print(f"图片真实标签为{tr_y},预测值为{pre_y},准确率为{acc}")
            # print(sess.run(features,feed_dict={data:np.expand_dims(test_set_x[nl],axis=0)}))
        # mcostime = sum(costime[1:])/(sts-1)
        # print(f'预测平均耗时为 {round(mcostime,6)} 秒')



def run():
    # 未修改的pb
    path = '01/tree_gai_merge_puring.pb'
    path = '01/tree_gai.pb'
    load_model(path)
    # tf.reset_default_graph()
    # # 修改的pb
    # path = '01/tree_gai_merge_puring_re.pb'
    # reload_model(path)




if __name__ == '__main__':
    run()