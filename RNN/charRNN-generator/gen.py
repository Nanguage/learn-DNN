import argparse

import chainer
import chainer.links as L
import numpy as np

from model import CharRNN
from utils import TextConverter


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True,
                        help='Path to model file name.')
    parser.add_argument('--converter', '-c',
                        help='Path to converter.pkl')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--n_hidden', '-n', type=int, default=1,
                        help='Number of hidden LSTM layer.')
    parser.add_argument('--start_string', '-s',
                        default='',
                        help='Start generation from this string.')
    parser.add_argument('--max_length', '-l', type=int, default=30,
                        help='Max length to generate.')
    return parser


def generate(model, converter, start_string, max_length):
    rnn = list(model.children())[0]
    if len(start_string) > max_length:
        start_string = start_string[:max_length]
    rnn.reset_state()

    np = rnn.xp

    result_arr = []
    for i in converter.text_to_arr(start_string):
        rnn(np.array([i]))
        result_arr.append(i)

    for _ in range(max_length - len(start_string)):
        if result_arr:
            i = result_arr[-1]
        else:
            i = 0
        y = rnn(np.array([i])).array[0]
        result_arr.append(int(np.argmax(y)))
    
    result_str = converter.arr_to_text(result_arr)

    return result_str


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    converter = TextConverter(filename=args.converter)
    rnn = CharRNN(converter.vocab_size, args.unit, n_hidden=args.n_hidden)
    model = L.Classifier(rnn)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    print(generate(model, converter, args.start_string, args.max_length))
