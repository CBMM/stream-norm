# Author: Qianli Liao
# 11/19/2016

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np


def py_func_with_grad(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+16))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# def update_streaming_variables(s_long,s_short,s_short_counter,s_w_updated_flag,alpha_or_beta,kappa):
#     if s_w_updated_flag.eval() != 0:
#         weight_updated = True
#     else:
#         weight_updated = False
#     if training_mode:
#         if weight_updated == True:
#             s_long.assign(s_long.eval()*kappa[1] + s_short.eval()*kappa[2])
#             s_short_counter.assign(0)
#         if s_short_counter == 0:
#             s_short.assign(x.eval())
#         else:
#             s_short.assign( s_short.eval()*s_short_counter.eval() + x.eval() )
#         s_short_counter.assign(s_short_counter.eval()+1)
#         s_short.assign(s_short.eval()/s_short_counter.eval())
#         s_used.assign(s_long*alpha_or_beta[1] + s_short*alpha_or_beta[2])


# v1 = get_scope_variable('foo', 'v', [1])
# v2 = get_scope_variable('foo', 'v')
# assert v1 == v2


def get_scope_variable(scope_name, var, **kwargs):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, **kwargs)
        except ValueError as e:
            scope.reuse_variables()
            v = tf.get_variable(var, **kwargs)
    return v



def get_streaming_variable(scope_name, var, shape, trainable, init_val=0):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'streaming_variables']
    return get_scope_variable(scope_name, var,
        shape=shape,
        initializer=tf.constant_initializer(init_val),
        collections=collections,
        trainable=trainable
    )

def update_streaming_variables(x,training_mode,v_long,v_short,v_used,v_short_counter,v_w_updated_flag,alpha_or_beta,kappa):
    first_update = (v_short_counter == -1)
    weight_updated = (v_w_updated_flag != 0)
    proceed = False
    if training_mode:
        #training_mode has higher priority
        proceed=True
    elif 'streaming_norm_training_mode_global_flag' in globals():
        #print 'check.......' 
        global streaming_norm_training_mode_global_flag
        if streaming_norm_training_mode_global_flag:
            proceed = True
    if proceed:
        #print 'snorm training.......'
        if weight_updated:
            #print 'weight updated.......'
            if first_update:
                v_long.assign(v_short)
            else:
                v_long.assign(v_long*kappa[0] + v_short*kappa[1])
            v_w_updated_flag.assign(0)
        if v_short_counter == 0:
            v_short.assign(x)
        else:
            v_short.assign( v_short*v_short_counter + x )
        v_short.assign(x)
        v_short_counter.assign(v_short_counter+1)
        #v_short.assign(v_short/v_short_counter)
        v_used.assign(v_long*alpha_or_beta[0] + v_short*alpha_or_beta[1])
        v_used.assign(v_short)


# def streamfb_internal(x, x_shape, name, training_mode, alpha, beta, kappa):
#     # aaa = tf.get_variable('aaa22',[])
#     #s_long  = get_streaming_variable(name,'s_long',shape=x.get_shape(),trainable=False)
#     # s_short = get_streaming_variable(name,'s_short',shape=x.get_shape(),trainable=False)
#     # s_used  = get_streaming_variable(name,'s_used',shape=x.get_shape(),trainable=False)
#     # s_short_counter  = get_streaming_variable(name,'s_short_counter',shape=[],trainable=False)
#     # s_w_updated_flag  = get_streaming_variable(name,'s_w_updated_flag',shape=[],trainable=True)
#     # update_streaming_variables(x,training_mode,s_long,s_short,s_used,s_short_counter,s_w_updated_flag,alpha[0:2],kappa[0:2])
#     # g_long  = get_streaming_variable(name,'g_long',shape=x.get_shape(),trainable=False)
#     # g_short = get_streaming_variable(name,'g_short',shape=x.get_shape(),trainable=False)
#     # g_used  = get_streaming_variable(name,'g_used',shape=x.get_shape(),trainable=False)
#     # g_short_counter  = get_streaming_variable(name,'g_short_counter',shape=[],trainable=False)
#     # g_w_updated_flag  = get_streaming_variable(name,'g_w_updated_flag',shape=[],trainable=True)    

#     s_long  = get_streaming_variable(name,'s_long',shape=x_shape,trainable=False)
#     s_short = get_streaming_variable(name,'s_short',shape=x_shape,trainable=False)
#     s_used  = get_streaming_variable(name,'s_used',shape=x_shape,trainable=False)
#     s_short_counter  = get_streaming_variable(name,'s_short_counter',shape=[],trainable=False)
#     s_w_updated_flag  = get_streaming_variable(name,'s_w_updated_flag',shape=[],trainable=True)
#     update_streaming_variables(x,training_mode,s_long,s_short,s_used,s_short_counter,s_w_updated_flag,alpha[0:2],kappa[0:2])
#     g_long  = get_streaming_variable(name,'g_long',shape=x_shape,trainable=False)
#     g_short = get_streaming_variable(name,'g_short',shape=x_shape,trainable=False)
#     g_used  = get_streaming_variable(name,'g_used',shape=x_shape,trainable=False)
#     g_short_counter  = get_streaming_variable(name,'g_short_counter',shape=[],trainable=False)
#     g_w_updated_flag  = get_streaming_variable(name,'g_w_updated_flag',shape=[],trainable=True)
#     #print s_used.eval()
#     print x*5
#     print s_used
#     return x*5


def streamfb(x, name, training_mode=False):
    alpha = [0,1]
    beta = [0,0,1]
    kappa = alpha * 2
    # if 'alpha_global' in globals():
    #     print 'alpha in globals'
    #     global alpha_global
    #     alpha = alpha_global
    # else:
    #     alpha = [0,1]
    # if 'beta_global' in globals():
    #     print 'beta in globals'
    #     global beta_global
    #     alpha = beta_global
    # else:
    #     beta = [0,0,1]
    # if 'kappa_global' in globals():
    #     print 'kappa in globals'
    #     global kappa_global
    #     kappa = kappa_global
    # else:
    #     kappa = alpha * 2
    #x_copy = tf.stop_gradient(x)
    x_shape = x.get_shape();
    s_long  = get_streaming_variable(name,'s_long',shape=x_shape,trainable=False)
    s_short = get_streaming_variable(name,'s_short',shape=x_shape,trainable=False)
    s_used  = get_streaming_variable(name,'s_used',shape=x_shape,trainable=False)
    s_short_counter  = get_streaming_variable(name,'s_short_counter',shape=[],trainable=False, init_val=-1)
    s_w_updated_flag  = get_streaming_variable(name, 's_w_updated_flag', shape=[], trainable=True, init_val=1)
    #s_w_updated_flag  = get_streaming_variable(name,'s_w_updated_flag',shape=[],trainable=False, init_val=1)
    update_streaming_variables(x,training_mode,s_long,s_short,s_used,s_short_counter,s_w_updated_flag,alpha[0:2],kappa[0:2])
    g_long  = get_streaming_variable(name,'g_long',shape=x_shape,trainable=False)
    g_short = get_streaming_variable(name,'g_short',shape=x_shape,trainable=False)
    g_used  = get_streaming_variable(name,'g_used',shape=x_shape,trainable=False)
    g_short_counter  = get_streaming_variable(name,'g_short_counter',shape=[],trainable=False, init_val=-1)
    g_w_updated_flag  = get_streaming_variable(name,'g_w_updated_flag',shape=[],trainable=True, init_val=1)
    #g_w_updated_flag  = get_streaming_variable(name,'g_w_updated_flag',shape=[],trainable=False, init_val=1)
    
    out = py_func_with_grad(lambda x: x,
        [x],
        [tf.float32],
        name=name,
        grad=lambda op,grad: streamfb_grad(op,grad,name,beta,kappa)
    )
    
    #out = tf.add(out, force_grad(s_w_updated_flag,1))
    #out = tf.add(out, force_grad(g_w_updated_flag,1))
    
    out = tf.reshape(out, x.get_shape())
    return out


# Actual gradient:
def streamfb_grad(op, grad, name, beta, kappa):
    x = op.inputs[0]
    #with tf.variable_scope(name,reuse=True):
    g_long  = get_streaming_variable(name,'g_long',shape=x.get_shape(),trainable=False)
    g_short = get_streaming_variable(name,'g_short',shape=x.get_shape(),trainable=False)
    g_used  = get_streaming_variable(name,'g_used',shape=x.get_shape(),trainable=False)
    g_short_counter  = get_streaming_variable(name,'g_short_counter',shape=[],trainable=False)
    g_w_updated_flag  = get_streaming_variable(name,'g_w_updated_flag',shape=[],trainable=True)
    update_streaming_variables(grad,True,g_long,g_short,g_used,g_short_counter,g_w_updated_flag,beta[0:2],kappa[2:4])
    #g_used.assign(g_used + beta[2]*grad)
    g_used.assign(grad)
    return tf.convert_to_tensor(g_used)  # to convert from float32_ref (mutable) to float32
    # return g_used


def force_grad(x,mag,name='force_grad'):
    # this op forwardprops 0, backprops mag
    out = py_func_with_grad(lambda x: np.float32(0),
                            [x],
                            [tf.float32],
                            name=name,
                            grad=lambda op, grad: force_grad_backprop(op,grad,mag) )
    return out

def force_grad_backprop(op,grad,mag):
    # backprop a constant gradient
    x = op.inputs[0]
    return x*0+mag



# global streaming_norm_training_mode_global_flag
# streaming_norm_training_mode_global_flag = True
# sess = tf.Session()
# #sess.as_default()
# #tf.initialize_all_variables().run()
# #with tf.Session() as sess:
# x = tf.constant([1., 2.])
# p = force_grad(x,100)
# y = streamfb(x,name='s1')
# tf.all_variables()
# sess.run(tf.initialize_all_variables())
# sess.run(p)
# sess.run(y)
# sess.run(tf.gradients(y, x))
# sess.run(tf.gradients(y, tf.trainable_variables()))



# with tf.Session() as sess:

