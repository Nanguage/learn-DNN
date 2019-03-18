# A Char-RNN chainer implementation

Char-RNN for text generation in chainer.

## Training

```
$ python train.py -i data/lingo.txt --gpu 0
```

## Generation

```
python gen.py -m model.npz -c converter.pkl -l 200


《郁罗症》/《AgUmy

Ofe it singing

我是这麼历上生满骛意全谎哭化的鲜丽


《祖国后》/《喧紙ル》


《赌博》/《ギャンブ》
我总是选择一切。

何处に月桃源郷全无いのな由、
她离很连这个夏日日摇的时到罢了
除了＂献身″外外外不会再来


《依存症》

忽然你是不说为吗
你是我的爱情吗？
我是我们的生命来
必经完全变这是比成理
终著我已紧不
```

## Reference

+ [Char-RNN tensorflow](https://github.com/hzy46/Char-RNN-TensorFlow)
+ [chainer RNN language model](https://docs.chainer.org/en/stable/examples/ptb.html)

