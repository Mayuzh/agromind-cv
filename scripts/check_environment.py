import json

import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"Build: {json.dumps(tf.sysconfig.get_build_info(), default=str)}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
