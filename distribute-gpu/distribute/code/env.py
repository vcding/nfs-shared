import os
import json

os.environ["TF_CONFIG"] = json.dumps({
'cluster': {
  'chief': ['10.244.0.15 :2222'],
  'ps': ['10.244.1.11:2222'],
  'worker': ['10.244.1.12:2222'],
  },
'task': {
  'type': "chief",
  'index': 0,
  }
})
