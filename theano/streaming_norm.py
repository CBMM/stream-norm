from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

from theano_sample import load_data, LogisticRegression, predict
from mlp_sample import HiddenLayer, MLP
import optimizers

def empty_shared(dims, name=None):
    if name:
        return theano.shared(value=np.zeros(dims,dtype=theano.config.floatX),name=name,borrow=True)
    return theano.shared(value=np.zeros(dims,dtype=theano.config.floatX),borrow=True)

def streaming_norm(params):
    learning_rate = params['learning_rate']
    n_epochs = params['n_epochs']
    dataset = params['dataset']
    batch_size = params['batch_size']
    update_interval = params['update_interval']
    short_long_rate = params['short_long_rate']
    L1_reg = params['L1_reg']
    L2_reg = params['L2_reg']
    n_hidden = params['n_hidden']

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    gparams = [T.grad(cost, param) for param in classifier.params]

    temp_params = []
    update_params

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    batch_update = [
        (t, t + g) for t, g in zip(temp_params, gparams)
    ]
    weight_update = [
        (p, p - learning_rate * (t + g) / update_interval) for p, t, g in zip(classifier.params, temp_params, gparams)
    ]

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    G_W = empty_shared((28 * 28, 10))
    G_b = empty_shared((10,))

    weight_update = [
        (classifier.W, classifier.W - learning_rate * (G_W + g_W) / update_interval),
        (classifier.b, classifier.b - learning_rate * (G_b + g_b) / update_interval),
        (G_W, empty_shared((28 * 28, 10))),
        (G_b, empty_shared((10,)))
    ]

    batch_update = [(G_W, G_W + g_W), (G_b, G_b + g_b)]

    train_model_with_update = theano.function(
        inputs=[index],
        outputs=cost,
        updates=weight_update,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model_without_update = theano.function(
        inputs=[index],
        outputs=cost,
        updates=batch_update,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training the model')
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % update_interval == 0:
                minibatch_avg_cost = train_model_with_update(minibatch_index)
            else:
                minibatch_avg_cost = train_model_without_update(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

if __name__ == '__main__':
    params = {
        'learning_rate':0.13,
        'n_epochs':1000,
        'dataset':'mnist.pkl.gz',
        'batch_size':50,
        'update_interval':4,
        'short_long_rate':0.7,
        'L1_reg':0.001,
        'L2_reg':0.001,
        'n_hidden':100
    }
    streaming_norm(params)

