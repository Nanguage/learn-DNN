import os
import argparse

import matplotlib
matplotlib.use("Agg")

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from model import CharRNN
from utils import TextConverter, ParallelSequentialIterator
from utils import BPTTUpdater, compute_perplexity

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', "-i", required=True,
                        help="Path to training text file.")
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--n_hidden', '-n', type=int, default=1,
                        help='Number of hidden LSTM layer.')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true', 
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Path to final model file name.')
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    model_path = os.path.dirname(args.model) or './'
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    with open(args.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text)
    print("#vovab={}".format(converter.vocab_size))
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    train = converter.text_to_arr(text)
    if args.test:
        train = train[:1000]
    train_iter = ParallelSequentialIterator(train, args.batchsize)

    rnn = CharRNN(converter.vocab_size, args.unit, n_hidden=args.n_hidden)
    model = L.Classifier(rnn)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/perplexity', 'main/accuracy']
    ), trigger=(interval, 'iteration'))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/accuracy'],
                                  'epoch', file_name='accuracy.png'))
   
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot(), trigger=(interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{}'.format(updater.iteration)
    ), trigger=(interval, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

    chainer.serializers.save_npz(args.model, model)
