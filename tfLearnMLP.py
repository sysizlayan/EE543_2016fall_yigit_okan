import tensorflow as tf
import cv2
import numpy as np
import tflearn
import matplotlib.pyplot as plt

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from dataRead import read_dataset,one_hot_to_index

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
    total_image1 = np.zeros((n_image, 100, 100, 1), dtype=np.uint8)
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
                total_image1[i] = example
            batchXX, batchYY, batchReal = sess.run([tf.constant(total_image), tf.constant(labels), tf.constant(total_image1)])
            coord.request_stop()
            coord.join(threads)
        return batchXX,batchYY, batchReal
dataset_location = "./firstDataSet"
number_of_images, number_of_training_images, number_of_validation_images, number_of_classes, classified_input_list, classified_validation_list, available_classes = read_dataset(dataset_location)#read the dataset

if (input_tf == "y" or input_tf == "Y"):
    write_tfrecords(classified_input_list, "./output/training-images/train")
    write_tfrecords(classified_validation_list, "./output/validation-images/valid")
else:
    print("====Skipping writing TF records====")

X, Y , RealIMG= get_all_records("./output/training-images/",number_of_training_images, number_of_classes)
test_x, test_y, asdn = get_all_records("./output/validation-images/",number_of_validation_images, number_of_classes)


X = X.reshape([-1, 10000])
test_x = test_x.reshape([-1, 10000])
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

model = tflearn.DNN(convnet, tensorboard_verbose=0,tensorboard_dir=".\\OUTMLP\\")
#model.fit({'input': X}, {'targets': Y}, n_epoch=100, snapshot_epoch=True, show_metric=True, run_id='caltechMLPHighEpoch1')
#model.fit({'input': X}, {'targets': Y}, validation_set=({'input': test_x}, {'targets': test_y}), n_epoch=200,snapshot_step=500, show_metric=True, run_id='caltech101MLPP')
#model.save('./MLPModel.tfl')
model.load('./MLPModel.tfl')

print("PREDICTION ", "\t\t\t", "ACTUAL")
# Compare original images with their reconstructions
for i in range(int(number_of_training_images)):
    prediction = np.round(model.predict([X[i]])[0])
    t=0
    for k in range(0,number_of_classes):
        if(prediction[k]==1):
            break
        else:
            t+=1

    predictionLabel = number_of_classes-t-1
    actual=Y[i]
    t = 0
    for k in range(0, number_of_classes):
        if (actual[k] == 1):
            break
        else:
            t += 1

    actualLabel = number_of_classes - t - 1
    print(available_classes[predictionLabel], "\t\t\t" ,available_classes[actualLabel])
# Compare original images with their reconstructions
'''f, a = plt.subplots(1, 10, figsize=(10, 2))
for i in range(10):
    prediction = np.round(model.predict([X[i]])[0])
    for sayi in range(number_of_classes):
        if(prediction[sayi]<0.01):
            prediction[sayi]=0.
        else:
            prediction[sayi]=1.
    prediction=np.array(prediction,dtype=np.int32)
    m = max(a)
    predictionLabel=number_of_classes-[i for i, j in enumerate(a) if j == m][0]-1
    #predictionLabel = one_hot_to_index(prediction,number_of_classes)
    actual=Y[i]
    actualLabel=one_hot_to_index(actual,number_of_classes)
    a[i].imshow(np.reshape(RealIMG[i], (100, 100)), cmap='gray')
    print(available_classes[predictionLabel])
f.show()
plt.draw()
plt.waitforbuttonpress()'''
