import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from dataRead import read_dataset

batch_size = 250
input_tf = input("Create TF Records? [y/n] ")

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
            image=sess.run(tf.cast(tf.image.decode_jpeg(image_file, channels=1),tf.uint8))
            image_bytes = image.tobytes()
            print(category, image.shape)
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


def get_all_records(FILE,n_image, n_class):

    total_image = np.zeros((n_image, 100, 100, 1),dtype=np.uint8)
    total_label = []
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(
                    FILE + "*.tfrecords"))
            image, label = read_and_decode(filename_queue)
            image = tf.reshape(image, tf.pack([100, 100, 1]))
            image.set_shape([100, 100, 1])
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            labels=[]
            for i in range(0, n_image):
                example, l = sess.run([(tf.cast(image, tf.float32)), label])
                tempLabel = (n_class - l - 1) * [0.] + [1.] + l * [0.]
                labels.append(tempLabel)
                total_image[i] = (np.ones((100,100,1),np.float32)-example/255)
            batchXX, batchYY = sess.run([tf.constant(total_image), tf.constant(labels)])
            coord.request_stop()
            coord.join(threads)
        return batchXX,batchYY
dataset_location = "./firstDataSet"
number_of_images, number_of_training_images, number_of_validation_images, number_of_classes, classified_input_list, classified_validation_list, dataset = read_dataset(dataset_location)#read the dataset

if (input_tf == "y" or input_tf == "Y"):
    write_tfrecords(classified_input_list, "./output/training-images/train")
    write_tfrecords(classified_validation_list, "./output/validation-images/valid")
else:
    print("====Skipping writing TF records====")

X, Y = get_all_records("./output/training-images/",number_of_training_images, number_of_classes)
#test_x, test_y = get_all_records("./output/validation-images/",number_of_validation_images, number_of_classes)


X = X.reshape([-1, 10000])

#test_x = test_x.reshape([-1, 10000])

convnet = input_data(shape=[None, 10000], name='input')

'''convnet = conv_2d(convnet, 24, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 96, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)'''


convnet = fully_connected(convnet, 1024, activation='relu')
convnet = fully_connected(convnet, 1024, activation='relu')
#convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, number_of_classes, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=1, checkpoint_path='./mldModel_CHPT')
model.fit({'input': X}, {'targets': Y}, n_epoch=100, snapshot_epoch=True, show_metric=True, run_id='caltechMLPHighEpoch1')
#model.fit({'input': test_x}, {'targets': test_y}, validation_set=({'input': X}, {'targets': Y}), n_epoch=30,snapshot_step=500, show_metric=True, run_id='caltechValid')
model.save('./asdmodel.tfl')

#model.restore('./asdmodel1.tfl')
#model.fit({'input': X}, {'targets': Y}, n_epoch=100, snapshot_epoch=True, show_metric=True, run_id='caltechMLPHighEpoch2')
#model.save('./asdmodel2.tfl')
print(np.round(model.predict(X[9:10]) [0]))

print(Y[0])