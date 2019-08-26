import tensorflow as tf

def rename_var(ckpt_path, new_ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            print(var_name)
            var = tf.contrib.framework.load_variable(ckpt_path, var_name)
            new_var_name1 = var_name.replace('Adam_1', '')
            var = tf.Variable(var, name=new_var_name1)
            new_var_name1 = var_name.replace('Adam','')
            var = tf.Variable(var, name=new_var_name1)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)

ckpt_path = '/Users/fengkai/PycharmProjects/tensorflow_classification/output/vgg16-9-0.96.ckpt-1270'
new_ckpt_path = '/Users/fengkai/PycharmProjects/tensorflow_classification/output/vgg16-0.96.ckpt'
rename_var(ckpt_path, new_ckpt_path)
