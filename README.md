# Graph_Nets_GCN

## Prerequisites
1. Linux or OSX
2. Python 3.6+
3. Sonnet(deepmind)
4. Tensorflow
5. Graphnet(deepmind)
## Usage
```
python Node_Apply_GCN.py  --epochs 500

```

## Paper
### Semi-Supervised Classification with Graph Convolutional Networks
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.

[[Paper]](https://arxiv.org/abs/1609.02907) [[Original Implementation]](https://github.com/tkipf/pygcn)

## Download  Dataset

["cora"](https://www.dropbox.com/s/3ggdpkj7ou8svoc/cora.zip?dl=1)


## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```

