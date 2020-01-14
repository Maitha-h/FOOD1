
import tensorflow as tf
import time
Start_time = time.time()
print("Sleeping")
time.sleep(600)
print(tf.test.is_gpu_available())
print("EXECUTION TIME: ", int((time.time() - Start_time) // 60), ":", int((time.time() - Start_time) % 60))