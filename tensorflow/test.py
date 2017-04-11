import tensorflow as tf

#from tensorflow.examples.tutorials.mnist import input_data

# 55000, 5000, 10000


# import streaming2

# # def get_scope_variable(scope_name, var, **kwargs):
# #     with tf.variable_scope(scope_name) as scope:
# #         try:
# #             v = tf.get_variable(var, **kwargs)
# #         except ValueError as e:
# #             scope.reuse_variables()
# #             v = tf.get_variable(var, **kwargs)
# #     return v

# # def get_scope_variable2(var, **kwargs):
    
# #     v = tf.get_variable(var, **kwargs)
# #     return v

# # with tf.variable_scope('wow') as scope:
# # 	#scope.reuse_variables()
# # 	a = get_scope_variable2('great', shape=[], initializer=tf.constant_initializer(0))
# # 	print a.name

# # var = streaming2.get_streaming_variable('scope', 'var', [2,3], trainable=True, init_val=1)
# # one = tf.constant(1.0)

# # equality = tf.equal(var, one)

# # true = tf.constant(True, dtype=tf.bool)
# # gg = tf.get_variable('namedd',initializer=tf.constant_initializer(True),shape=[],dtype=tf.bool)
# # a = gg.assign(true)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)

# 	print sess.run(a)


x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

# with tf.control_dependencies([x_plus_1]):
#     y = tf.identity(x_plus_1)

y = x_plus_1
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for i in xrange(5):
        print(y.eval())