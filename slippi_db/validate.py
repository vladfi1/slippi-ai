from slippi_db import upload_lib

s3 = upload_lib.s3
bucket = s3.bucket

def validate_s3(env, stage, key):
  path = '/'.join([env, stage, key])
  object = bucket.Object(path)
  return object.content_length > 0

def size_to_str(size: int):
  return '%.3f' % (size / 10 ** 9)

def validate(env: str, stage):
  db = upload_lib.get_db(env, stage)
  lost = upload_lib.db.get_collection('-'.join([env, stage, 'lost']))

  invalid = []
  for doc in db.find({}):
    key = doc['key']
    valid = validate_s3(env, stage, key)
    if not valid:
      print(
        doc['filename'],
        doc['description'],
        size_to_str(doc['stored_size']))
      invalid.append(doc)
      db.delete_one({'key': key})

    # db.update_one({'key': key}, {'$set': {'valid': valid}})
  
  lost.insert_many(invalid)

if __name__ == '__main__':
  validate('prod', 'raw')
