
import os, argparse, time
import threading
import queue
from typing import Optional

import dropbox, dropbox.files

def format_bytes(bytes_value):
  """Convert bytes to human readable format"""
  for unit in ['B', 'KB', 'MB', 'GB']:
    if bytes_value < 1024.0:
      return f"{bytes_value:.1f} {unit}"
    bytes_value /= 1024.0
  return f"{bytes_value:.1f} TB"

def format_time(seconds):
  """Format seconds into MM:SS or HH:MM:SS"""
  if seconds < 3600:
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"
  else:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def upload_worker(
    dbx: dropbox.Dropbox,
    chunk_queue: queue.Queue,
    session_id: str,
    worker_id: int,
    stop_event: threading.Event
):
  """Worker thread that uploads chunks from the queue"""
  cursor = None

  while not stop_event.is_set():
    try:
      # Get chunk data and offset
      chunk_data, chunk_offset, is_last = chunk_queue.get(timeout=1.0)

      cursor = dropbox.files.UploadSessionCursor(
        session_id=session_id,
        offset=chunk_offset,
      )

      # Upload the chunk
      dbx.files_upload_session_append_v2(chunk_data, cursor, close=is_last)

      chunk_queue.task_done()

    except queue.Empty:
      continue
    except Exception as e:
      print(f"\nWorker {worker_id} error: {e}")
      break

def upload_file(
    token: str,
    file_path: str,
    upload_folder: str,
    chunk_size_mb: int = 8,
    num_workers: int = 4,
):
  dbx = dropbox.Dropbox(token)
  file_size = os.path.getsize(file_path)
  CHUNK_SIZE = chunk_size_mb * (2 ** 20)
  dest_path = upload_folder + '/' + os.path.basename(file_path)
  start_time = time.time()

  print(f"Uploading {os.path.basename(file_path)} ({format_bytes(file_size)})")
  print(f"Using {num_workers} worker threads with {chunk_size_mb}MB chunks")
  print("=" * 60)

  with open(file_path, 'rb') as f:
    uploaded_size = 0
    last_update_time = start_time

    # Small file - single upload
    if file_size <= CHUNK_SIZE:
      dbx.files_upload(f.read(), dest_path)
      total_time = time.time() - start_time
      transfer_rate = file_size / total_time if total_time > 0 else 0

      print(f"\r[{'█' * 50}] 100% | {format_bytes(transfer_rate)}/s | Time: {format_time(total_time)}")
      print(f"\nUpload completed in {format_time(total_time)}")
      return

    # Large file - concurrent chunked upload
    upload_session_start_result: dropbox.files.UploadSessionStartResult
    upload_session_start_result = dbx.files_upload_session_start(  # type: ignore[reportAssignmentType]
      b'',
      session_type=dropbox.files.UploadSessionType.concurrent,
    )

    # Initialize queues and synchronization
    chunk_queue = queue.Queue(maxsize=num_workers * 2)  # Buffer for chunks
    stop_event = threading.Event()

    # Start worker threads
    workers = []
    for i in range(num_workers):
      worker = threading.Thread(
        target=upload_worker,
        args=(dbx, chunk_queue, upload_session_start_result.session_id, i, stop_event)
      )
      worker.daemon = True
      worker.start()
      workers.append(worker)

    # Producer: read chunks and add to queue
    uploaded_size = 0

    try:
      while f.tell() < file_size:
        # Read chunk and add to queue
        chunk_offset = f.tell()
        chunk_data = f.read(CHUNK_SIZE)
        uploaded_size += len(chunk_data)
        is_last = uploaded_size == file_size
        chunk_queue.put((chunk_data, chunk_offset, is_last))

        # Update progress
        current_time = time.time()
        time_since_last_update = current_time - last_update_time
        if time_since_last_update >= 0.5:  # Update every 0.5 seconds
          elapsed_time = current_time - start_time
          transfer_rate = uploaded_size / elapsed_time

          progress = min(100.0, (uploaded_size / file_size) * 100)
          progress_bars = int((progress / 100) * 50)

          # Calculate ETA
          remaining_bytes = file_size - uploaded_size
          eta_seconds = remaining_bytes / transfer_rate if transfer_rate > 0 else 0

          print(f"\r[{'█' * progress_bars}{' ' * (50 - progress_bars)}] {progress:.1f}% | {format_bytes(transfer_rate)}/s | ETA: {format_time(eta_seconds)}", end='', flush=True)

          last_update_time = current_time

      assert uploaded_size == file_size, f"Uploaded size {uploaded_size} != file size {file_size}"

      # Final chunk - wait for all chunks to finish, then finish session
      chunk_queue.join()  # Wait for all chunks to be processed
      stop_event.set()

      final_cursor = dropbox.files.UploadSessionCursor(
        session_id=upload_session_start_result.session_id,
        offset=uploaded_size,
      )
      commit = dropbox.files.CommitInfo(path=dest_path)
      dbx.files_upload_session_finish(b'', final_cursor, commit)

      # Final progress update
      total_time = time.time() - start_time
      transfer_rate = uploaded_size / total_time if total_time > 0 else 0
      progress = min(100.0, (uploaded_size / file_size) * 100)

      print(f"\r[{'█' * 50}] {progress:.1f}% | {format_bytes(transfer_rate)}/s | Time: {format_time(total_time)}")
      print(f"\nUpload completed in {format_time(total_time)}")

    finally:
      # Clean up worker threads
      stop_event.set()
      for worker in workers:
        worker.join(timeout=1.0)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, help='path to file to upload')
  parser.add_argument('--upload_path', type=str, help='path in dropbox')
  parser.add_argument('--token', type=str, help='dropbox token', default=os.getenv('DBOX_ACCESS_TOKEN'))
  parser.add_argument('--chunk_size_mb', type=int, default=8, help='chunk size in MB')
  parser.add_argument('--workers', type=int, default=4, help='number of worker threads for concurrent uploads')
  args = parser.parse_args()

  if args.token is None:
    raise ValueError('No token provided!')

  upload_file(
      token=args.token,
      file_path=args.file_path,
      upload_folder=args.upload_path,
      chunk_size_mb=args.chunk_size_mb,
      num_workers=args.workers,
  )

if __name__ == "__main__":
  main()