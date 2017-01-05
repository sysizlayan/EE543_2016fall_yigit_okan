import tensorflow as tf
import numpy as np
from dataRead import read_dataset

input_tf = input("Create TF Records? [y/n] ")

batch_size = 250

def write_tfrecords(classified_list, output_location, batch_size=batch_size):
    iteration=0
    writer = None
    cur_index = 0
    with tf.Session() as sess:
        for category, image_filename in classified_list:
            print("Category: ", category)
            print("File: ", image_filename)
            if cur_index % batch_size == 0: # 100 by default
                if writer:
                    writer.close() # No idea what this does ;)
                record_filename = "{a}-{b}.tfrecords".format(a=output_location, b=iteration)
                print(record_filename)
                fw = open(record_filename, "w")
                fw.close()
                writer = tf.python_io.TFRecordWriter(record_filename) # Creates a TFRecord per batch
                print("Generated", record_filename)
                iteration += 1
            cur_index += 1
            image_file = tf.read_file(image_filename)
            image_bytes = sess.run(tf.image.decode_jpeg(image_file, channels=1)).tobytes()
            print(category)
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[category])),
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized_example,
        features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    return image, label


def get_all_records(FILE,n_batch, n_class):

    total_image = np.zeros((n_batch, 10000),dtype=np.uint8)
    total_label = []
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(
                    FILE + "*.tfrecords"))
            image, label = read_and_decode(filename_queue)
            image = tf.reshape(image, tf.pack([1, 10000]))
            image.set_shape([1, 10000])
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            labels=[]
            for i in range(0, n_batch):
                example, l = sess.run([tf.cast(image, tf.uint8), label])
                tempLabel = (n_class - l - 1) * [0] + [1] + l * [0]
                labels.append(tempLabel)
                total_image[i] = example
            batchXX, batchYY = sess.run([tf.constant(total_image), tf.constant(labels)])
            coord.request_stop()
            coord.join(threads)
        return batchXX,batchYY
dataset_location = ".\\firstDataSet"
number_of_images, number_of_training_images, number_of_validation_images, number_of_classes, classified_input_list, classified_validation_list, dataset = read_dataset(dataset_location)#read the dataset

if (input_tf == "y" or input_tf == "Y"):
    write_tfrecords(classified_input_list, ".\\output\\training-images\\train")
    write_tfrecords(classified_validation_list, ".\\output\\validation-images\\valid")
else:
    print("====Skipping writing TF records====")
    
dataX, dataY = get_all_records(".\\output\\training-images\\",number_of_training_images, number_of_classes)
validX, validY = get_all_records(".\\output\\validation-images\\",number_of_validation_images, number_of_classes)
print("DATALAR ALINDIII!")

# Parameters
learning_rate = 0.001
training_epochs = 100
display_step = 1

# Network Parameters
n_hidden_1 = 2000 # 1st layer number of features
n_hidden_2 = 2000 # 2nd layer number of features
n_input = 100*100 # MNIST data input (img shape: 28*28)
n_classes =  number_of_classes# MNIST total classes (0-9 digits)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.device("/gpu:0"):
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            #avg_cost = 0.
            #total_batch = int(number_of_training_images/batch_size)

            batch_x = dataX
            batch_y = dataY
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            #avg_cost += c

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: validX, y: validY}))
