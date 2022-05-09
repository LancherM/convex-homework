import numpy
import scipy.special
import matplotlib.pyplot

# 各种节点的个数
input_nodes = 784
hidden1_nodes = 100
hidden2_nodes = 200
output_nodes = 10

learning_rate = 0.1


# 读取数据集
def read_dataset(filename):
    data_file = open(filename, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list


class neuralNetwork:
    """初始化神经网络
    总共四层神经元
    分别是输入层，两个隐藏层和输出层
    """
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # 各种节点
        self.inodes = inputnodes
        self.h1nodes = hiddennodes1
        self.h2nodes = hiddennodes2
        self.onodes = outputnodes

        # 学习率
        self.lr = learningrate

        # 随机生成权重矩阵
        self.wih1 = (numpy.random.rand(self.h1nodes, self.inodes) - 0.5)
        self.wh1h2 = (numpy.random.rand(self.h2nodes, self.h1nodes) - 0.5)
        self.wh2o = (numpy.random.rand(self.onodes, self.h2nodes) - 0.5)

        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转化为向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算第一个隐藏层
        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # 计算第二个隐藏层
        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # 计算输出层
        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.wh2o.T, output_errors)
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)

        # 根据误差调整权重
        self.wh2o += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden2_outputs))
        self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1 - hidden1_outputs)),
                                        numpy.transpose(inputs))
        self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1 - hidden2_outputs)),
                                         numpy.transpose(hidden1_outputs))
        pass

    # 查询神经网络
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    n = neuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)
    # 导入mnist训练集
    dataset = read_dataset('mnist_train.csv')

    # 训练
    for data in dataset:
        value = data.split(',')
        inputs = (numpy.asfarray(value[1:])/255.0*0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(value[0])] = 0.99
        n.train(inputs, targets)

    # 测试训练结果
    testset = read_dataset('mnist_train_100.csv')
    testcard = []
    for data in testset:
        value = data.split(',')
        answer = int(value[0])
        inputs = (numpy.asfarray(value[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        if label == answer:
            testcard.append(1)
        else:
            testcard.append(0)

    # 输出训练正确率
    test_array = numpy.asarray(testcard)
    print(test_array.sum()/test_array.size)

