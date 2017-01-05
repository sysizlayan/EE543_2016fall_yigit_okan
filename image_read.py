import tensorflow as tf
import numpy as np
from dataRead import read_dataset
#import cv2

# read_image() is no longer necessary
# write_tfrecords() does the same thing

def read_image(filenames):
    #print(filename)
    total_image=[]
    hasItbegin=0
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            for i in filenames:
                print(hasItbegin)
                image_file = tf.read_file(i)
                if (hasItbegin == 0):
                    total_image = sess.run(tf.reshape(tf.cast(tf.image.decode_jpeg(image_file, channels=1), tf.float32), [1, 100 * 100]))
                else:
                    image = sess.run(tf.reshape(tf.cast(tf.image.decode_jpeg(image_file, channels=1), tf.float32), [1, 100 * 100]))
                    total_image = sess.run(tf.concat(0, [total_image, image]))
                hasItbegin += 1

    return total_image


"""
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            for i in filename:
                image_file = tf.read_file(i)
                try:
                    image = sess.run(tf.cast(tf.image.decode_jpeg(image_file,channels=1),tf.float32))
                except:
                    print(i)

                #print(tf.Session().run(tf.shape(image)))
                image1=sess.run(tf.reshape(image,[1, 100*100]))
                if (hasItbegin==0):
                    total_image=image1
                    hasItbegin=1
                else:
                    total_image = sess.run(tf.concat(0, [total_image, image1]))


            size = sess.run(tf.shape(total_image))
            print(size)
    return total_image"""

def write_tfrecords(classified_list, output_location, batch_size=100):
    writer = None
    cur_index = 0
    with tf.Session() as sess:
        for category, image_filename in classified_list:
            print("Category: ", category)
            print("File: ", image_filename)
            if cur_index % batch_size == 0: # 100 by default
                if writer:
                    writer.close() # No idea what this does ;)
                record_filename = "{a}-{b}.tfrecords".format(a=output_location, b=cur_index)
                print(record_filename)
                fw = open(record_filename, "w")
                fw.close()
                writer = tf.python_io.TFRecordWriter(record_filename) # Creates a TFRecord per batch
                print("Generated", record_filename)
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

### THE FOLLOWING TO BE REMOVED ###
"""
def write_records_file(dataset, record_location):

    #Fill a TFRecords file with the images found in `dataset` and include their category.
    #Parameters
    #----------
    #dataset : dict(list)
    #Dictionary with each key being a label for the list of image filenames of its value.
    #record_location : str
    #Location to store the TFRecord output.

    writer = None
    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
            record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,current_index=current_index)
            writer = tf.python_io.TFRecordWriter(record_filename)
        current_index += 1
        image_file = tf.read_file(image_filename)

        # In ImageNet dogs, there are a few images which TensorFlow doesn't recognize as JPEGs. This
        # try/catch will ignore those images.
        try:
            image = tf.image.decode_jpeg(image_file)
        except:
            print(image_filename)
            continue
        # Converting to grayscale saves processing and memory but isn't required.
        grayscale_image = tf.image.rgb_to_grayscale(image)
        resized_image = tf.image.resize_images(grayscale_image, 250, 151)
        # tf.cast is used here because the resized images are floats but haven't been converted into
        # image floats where an RGB value is between [0,1).
        image_bytes = tf.Session().run(tf.cast(resized_image, tf.uint8)).tobytes()
        # Instead of using the label as a string, it'd be more efficient to turn it into either an
        # integer index or a one-hot encoded rank one tensor.
        # https://en.wikipedia.org/wiki/One-hot
        image_label = breed.encode("utf-8")
        example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))
    writer.write(example.SerializeToString())
    writer.close()
write_records_file(testing_dataset, "./output/testing-images/testing-image")
write_records_file(training_dataset, "./output/training-images/training-image")
"""


def load_tfrecords():
    config = tf.ConfigProto(allow_soft_placement=True)
    sess=tf.Session(config=config)

    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("C:\\Users\\Yigit\\Desktop\\git\\EE543_SemesterProject_Okan_Yigit\\data_set_reading\\output\\training-images\\train-0.tfrecords"))
    reader = tf.TFRecordReader()

    _, serialized = reader.read(filename_queue)
    print("OKUYOM BEN YAA!")
    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        })
    print("HALA OKUYOM BEN YAA!")

    raw_image=tf.decode_raw(features['image'], tf.uint8)
    image=tf.reshape(raw_image,[100,100,1])
    label=features['label']

    print("OKUMAYA BAŞLADI")
    im,lab=sess.run([image,label])
    print(label)
    print("SANIRIM OKUDUM")
    """min_after_dequeue = 10
    batch_size = 3
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)"""

    #return sess.run([image_batch,label_batch])

### Debug part ###
#number_of_images, number_of_training_images, number_of_validation_images, number_of_classes, classified_input_list, classified_validation_list, dataset= read_dataset(".\\firstDataSet")
#write_tfrecords(classified_input_list, "C:\\Users\\Yigit\\Desktop\\git\\EE543_SemesterProject_Okan_Yigit\\data_set_reading\\output\\training-images\\train")
#write_tfrecords(classified_validation_list, "C:\\Users\\Yigit\\Desktop\\git\\EE543_SemesterProject_Okan_Yigit\\data_set_reading\\output\\validation-images\\valid")
#print("TFRecords have been generated.")
#load_tfrecords()
