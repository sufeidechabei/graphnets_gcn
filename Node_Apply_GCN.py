import sonnet as snt
import tensorflow as tf
import argparse
import numpy as np
from graph_nets import blocks
from graph_nets.modules import GraphIndependent
from graph_nets.blocks import EdgeBlock, NodeBlock
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.demos import models
from graph_nets import modules
from utils import load_data
import time
import graph_nets


parser = argparse.ArgumentParser(description='GCN')
parser.add_argument("--gpu", type=int, default=-1,
                    help="gpu")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--hidden_units", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--epoch", type=int, default=500)
args = parser.parse_args()

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    'cora')
w = np.where(train_mask, 1, 0)
num_classes = y_train.shape[1]
input_graphs = []
adj = adj.toarray()
index = np.where(adj != 0)
edges = tf.constant(np.concatenate(index).reshape(-1, 2))
senders = tf.constant(index[0], dtype=tf.int32)
receivers = tf.constant(index[1], dtype=tf.int32)


def make_linear_model():
    if args.layer == 1:
        return snt.Sequential(
            [snt.Linear(num_classes), ReLu()])
    else:
        layer_list = []
        layer_list.append(snt.Linear(args.hidden_units))
        layer_list.append(ReLu())

    return snt.Sequential(layer_list)


def edge_model_fn():
    return snt.Sequential([edgeclass()])


def dropout_fn():
    return snt.Sequential([Dropout(args.dropout)])


def final_layer():
    return snt.Sequential([snt.Linear(num_classes), ReLu()])


# define the dropout layer
class Dropout(snt.AbstractModule):
    def __init__(self, dropout, name='sntdropout'):
        super(Dropout, self).__init__(name=name)
        self.dropout = dropout

    def _build(self, inputs):
        print("dropout")
        return tf.nn.dropout(inputs, 1 - self.dropout)


# define the activate function.
class ReLu(snt.AbstractModule):
    def __init__(self, name="sntrelu"):
        super(ReLu, self).__init__(name=name)
        pass

    def _build(self, inputs):
        print("relu")
        return tf.nn.relu(inputs)

# define the edge class


class edgeclass(snt.AbstractModule):
    def __init__(self, name="edgeclass"):
        super(edgeclass, self).__init__(name=name)

    def _build(self, inputs):
        print("edge")
        return inputs

# define the GCN module
# message passing is included in the following code (Edgeblock, Nodeblock)


class GCN(snt.AbstractModule):
    def __init__(self, layers=args.layer, name="gcn"):
        super(GCN, self).__init__(name=name)
        with self._enter_variable_scope():
            self._networks = []
            if layers == 1:
                self._networks.append(
                    GraphIndependent(
                        node_model_fn=dropout_fn))
                self._networks.append(
                    EdgeBlock(
                        edge_model_fn=edge_model_fn,
                        use_edges=False,
                        use_receiver_nodes=True,
                        use_sender_nodes=False,
                        use_globals=False))
                self._networks.append(
                    NodeBlock(
                        node_model_fn=make_linear_model,
                        use_received_edges=True,
                        use_sent_edges=False,
                        use_nodes=False,
                        use_globals=False))
            else:
                for layer in range(args.layer - 1):
                    self._networks.append(
                        GraphIndependent(
                            node_model_fn=dropout_fn))
                    self._networks.append(
                        EdgeBlock(
                            edge_model_fn=edge_model_fn,
                            use_edges=False,
                            use_receiver_nodes=True,
                            use_sender_nodes=False,
                            use_globals=False))
                    self._networks.append(
                        NodeBlock(
                            node_model_fn=make_linear_model,
                            use_received_edges=True,
                            use_sent_edges=False,
                            use_nodes=False,
                            use_globals=False))
                self._networks.append(
                    GraphIndependent(
                        node_model_fn=dropout_fn))
                self._networks.append(
                    EdgeBlock(
                        edge_model_fn=edge_model_fn,
                        use_edges=False,
                        use_receiver_nodes=True,
                        use_sender_nodes=False,
                        use_globals=False))
                self._networks.append(
                    NodeBlock(
                        node_model_fn=final_layer,
                        use_received_edges=True,
                        use_sent_edges=False,
                        use_nodes=False,
                        use_globals=False))

    def _build(self, inputs):
        for layer in self._networks:
            inputs = layer(inputs)
        return inputs


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


x = tf.placeholder(shape=[None, features.shape[1]], dtype=tf.float32)
y = tf.placeholder(shape=[None, y_train.shape[1]], dtype=tf.float32)
train_mask_holder = tf.placeholder(shape=[None, ], dtype=tf.int32)

# define a Graph object
input_graphs.append({"nodes": x,
                     "edges": edges,
                     "receivers": receivers,
                     "senders": senders})
input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_graphs)
model = GCN()


output_graphs = model(input_graphs)

# Make the graph can be ran in a session

# Define the loss function
output_graph = utils_tf.get_graph(output_graphs, 0)
output_graphs, input_graphs, output_graph = make_all_runnable_in_session(
    output_graphs, input_graphs, output_graph)
loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.boolean_mask(
        y, train_mask_holder), logits=tf.boolean_mask(
            output_graph.nodes, train_mask_holder))

# define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
shape = train_mask.shape[0]
step_op = optimizer.minimize(loss)
sess = tf.Session()

# initialize all the variables in the graph.
sess.run(tf.global_variables_initializer())
# run the GCN model in GPU.
with tf.device('/gpu:0'):
    t0 = time.time()
    for epoch in range(args.epoch):
        loss_data, _ = sess.run([loss, step_op], feed_dict={
                                x: features, y: y_train, train_mask_holder: train_mask})
        print(
            "| Epoch {:3d} | Loss is {:5.5f}".format(
                epoch + 1,
                loss_data.sum() / shape))

    t_end = time.time()
    print("The graph_net per epoch time is " + str((t_end - t0) / args.epoch))
