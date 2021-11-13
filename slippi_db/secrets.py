import os
import json

SECRETS = {}

SECRETS_FILE = os.path.expanduser('~/secrets.json')
if os.path.exists(SECRETS_FILE):
  with open(SECRETS_FILE) as f:
    SECRETS.update(json.load(f))

SECRET_KEYS = ['S3_CREDS', 'MONGO_URI']
for key in SECRET_KEYS:
  if key in os.environ:
    SECRETS[key] = os.environ[key]

for key in SECRET_KEYS:
  assert key in SECRETS, f'{key} not set'
