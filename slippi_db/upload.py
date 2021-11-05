import os

from flask import app, Flask, request

import upload_lib

replay_db = upload_lib.ReplayDB(upload_lib.NAME)
app = Flask(upload_lib.NAME)

home_html = """
<html>
  Upload a collection ({extensions}) of replays.
  <br/>
  Currently have {num_mb} MB uploaded to database "{db}".
  <br/>
  <body>
    <form action = "/upload" method = "POST" enctype = "multipart/form-data">
      <p><input type = "file" name = "file" /></p>
      <p>Description: <input type = "text" name = "description" /></p>
      <p><input type = "submit"/></p>
    </form>
  </body>
</html>
"""

RAW_EXTENSIONS = ('zip', '7z')

@app.route('/')
def homepage():
  return home_html.format(
    extensions='/'.join(RAW_EXTENSIONS),
    num_mb=replay_db.raw_size() // upload_lib.MB,
    db=upload_lib.NAME,
  )

@app.route('/upload', methods = ['POST'])
def upload_file():
  f = request.files['file']
  extension = f.filename.split('.')[-1]
  if extension in RAW_EXTENSIONS:
    # return replay_db.upload_zip(f)
    return replay_db.upload_raw(
      f, obj_type=extension, description=request.form['description'])
  else:
    return f'{f.filename}: must be in {RAW_EXTENSIONS}'

if __name__ == '__main__':
  # app.run(host='0.0.0.0', debug=False)
  app.run(debug=False)
