import os
import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.examples.tutorials.mnist import input_data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
def get_mnist():
    def load_data():
        mnist = input_data.read_data_sets(r'C:\Users\Cheng\.keras\datasets\t10k-images-idx3-ubyte', one_hot=True)
        return mnist

    mnist = load_data()
    test_set_x = mnist.test.images
    test_set_y = mnist.test.labels
    return test_set_x, test_set_y
# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='01/tree_gai_merge_puring_quantize_frozen_graph.tflite')
interpreter.allocate_tensors()
# 获取输入输出的张量

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


test_set_x, test_set_y = get_mnist()
batch_size = 1
n_test_batches = test_set_x.shape[0]
n_test_batches = int(n_test_batches / batch_size / 20)
costime=[]
total_predictiong=[]
for i in range(n_test_batches):
    a = time.time()
    data = np.expand_dims(test_set_x[i],axis=0)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    feature_data = interpreter.get_tensor(output_details[0]['index'])
    correct_prediction = np.equal(np.argmax(feature_data, 1), np.argmax(np.expand_dims(test_set_y[i],axis=0), 1))
    b=time.time()
    costime.append(b - a)
    total_predictiong.append(correct_prediction)



accuracy = np.mean(total_predictiong)
print("acc: ", accuracy)
print("costime : ",sum(costime[1:])/(n_test_batches-1))
print("costime : ",sum(costime)/(n_test_batches))