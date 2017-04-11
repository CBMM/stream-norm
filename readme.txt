The code is in "tensorflow" folder

The paper is:
https://arxiv.org/abs/1610.06160

TF: TensorFlow

The main question I'd like to figure out:
In tensorflow/streaming.py, I define a new OP (TF opearation) in the streaming function:

py_func_with_grad(lambda x, s: s, [x, s_final], [tf.float32], name=name, grad=lambda op,grad: stream_gradient_backprop(op,grad, scope.name ,beta,kappa))

The forward part of this OP is identity: f(x,s) = s, I use two parameters, since I want the gradient goes to x, instead of s. But I want to use the value of s in the forward prop.

The backprop part of this OP is defined in the function:  stream_gradient_backprop: It gets some TF variables by tf.get_variable, modify them in some way and store them back (using "update_streaming" function). I am not sure how TensorFlow deal with the computations defined in this backprop function. 

To run the code involving streaming normalization:
python mnist.py --hidden 100 --cell_type SNGRU --dau 20 --batch_size 5

To run the baseline code:
python mnist.py --hidden 100 --cell_type LNGRU --dau 20 --batch_size 5


The streaming normalization should work at least as well as the baseline. But since I do not know what TensorFlow does in the backprop part, I cannot debug..  


