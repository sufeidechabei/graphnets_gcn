import sonnet as snt
import tensorflow as tf
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
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
input_graphs = {}
input_graphs.append({"nodes":features})
input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_graphs)
input_graphs = utils_tf.fully_connect_graph_dynamic(input_graphs)
adj = tf.constant(adj)
def make_linear_model():
    
class GCN(snt.AbstractModule):
    def __init__(self, name = "GCNmodule"):
        with self._enter_variable_scope():
























