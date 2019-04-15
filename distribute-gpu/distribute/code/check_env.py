import os
import json

tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
print("TF_CONFIG env variable: %s", tf_config)
print("Node type is: %s", tf_config['task']['type'])
