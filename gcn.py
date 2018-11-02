import sonnet as snt
import tensorflow as tf
import argparse
import numpy as np
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.demos import models
from graph_nets import modules
from utils import load_data
import graph_nets
'''
a = tf.constant([0.5088539, 0.5530578, 0.15916789])




indice = tf.contrib.framework.argsort(a, -1)
rank = tf.contrib.framework.argsort(indice, -1)/2
sess = tf.Session()
x = tf.stack((a[:-1], a[1:]))[None]
x_shape = tf.shape(x)
si = tf.cast(tf.squeeze(a), tf.int32)
nodes = tf.one_hot(si[:1], 3)
node_shape = tf.shape(nodes)
shape = tf.shape(a[:, None])
input_graphs = []
input_graphs.append({"nodes":a[:, None]})
input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_graphs)
input_graphs = utils_tf.fully_connect_graph_dynamic(input_graphs)
input_graph = utils_tf.get_graph(input_graphs, 0)
senders = input_graph.senders
receivers = input_graph.receivers
stack_sender_receiver = tf.stack((senders, receivers), axis=1)
stack_again = tf.shape(stack_sender_receiver[:, :, None])
print(sess.run([senders, receivers, stack_sender_receiver, stack_again, x_shape]))
nodes = tf.constant(np.array([[0], [0], [0], [0]]))
edges = tf.constant(np.array([[0]]))
graph = input_graph.replace(nodes=nodes, edges=edges)
receivers = blocks.broadcast_receiver_nodes_to_edges(graph)
senders = blocks.broadcast_sender_nodes_to_edges(graph)
print(sess.run([receivers, senders]))
'''
parser = argparse.ArgumentParser(description='GCN')
parser.add_argument("--gpu", type=int, default=-1,
                    help="gpu")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--hidden_units", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--epoch", type=int, default=19)
args = parser.parse_args()
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
print(type(train_mask))
w = np.where(train_mask==True, 1, 0)
num_classes = y_train.shape[1]
input_graphs = []
x = tf.placeholder(shape = [None, features.shape[1]], dtype=tf.float32)
y = tf.placeholder(shape = [None, y_train.shape[1]], dtype=tf.float32)
train_mask_holder = tf.placeholder(shape = [None, ], dtype=tf.int32)
input_graphs.append({"nodes":x})
input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_graphs)
input_graphs = utils_tf.fully_connect_graph_dynamic(input_graphs)
adj = tf.constant(adj.toarray(), dtype=tf.float32)
def make_linear_model():
    if args.layer==1:
        return snt.Sequential([Dropout(args.dropout), snt.Linear(num_classes), MultiAdj(), ReLu()])
    else:
        layer_list = []
        for layer in range(args.layer-1):
            layer_list.append(Dropout(args.dropout))
            layer_list.append(snt.Linear(num_classes))
            layer_list.append(MultiAdj())
            layer_list.append(ReLu())



class MultiAdj(snt.AbstractModule):
    def __init__(self, name = "MultiAdj"):
        super(MultiAdj, self).__init__(name=name)
        pass
    def _build(self, inputs):
        print("mult")
        return tf.matmul(adj, inputs)


class Dropout(snt.AbstractModule):
    def __init__(self, dropout, name='sntdropout'):
        super(Dropout, self).__init__(name=name)
        self.dropout = dropout
    def _build(self, inputs):
        print("dropout")
        return tf.nn.dropout(inputs, 1 -self.dropout)

class ReLu(snt.AbstractModule):
    def __init__(self, name = "sntrelu" ):
        super(ReLu, self).__init__(name=name)
        pass
    def _build(self, inputs):
        print("relu")
        return tf.nn.relu(inputs)

class GCN(snt.AbstractModule):
    def __init__(self, name="gcn"):
        super(GCN, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(node_model_fn=make_linear_model)

    def _build(self, inputs):
        return self._network(inputs)

def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]

model = GCN()
input_graph = utils_tf.get_graph(input_graphs, 0)
output_graph = model(input_graph)
output_graph, input_graph = make_all_runnable_in_session(output_graph, input_graph)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(y, train_mask_holder),
                                               logits=tf.boolean_mask(output_graph.nodes, train_mask_holder))

optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
shape = train_mask.shape[0]
step_op = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(args.epoch):
    loss_data, _ = sess.run([loss, step_op], feed_dict={x:features, y:y_train, train_mask_holder:train_mask})
    print("| Epoch {:3d} | Loss is {:5.5f}".format(epoch + 1, loss_data.sum()/shape))

















































