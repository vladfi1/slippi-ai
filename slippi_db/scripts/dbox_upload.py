
import os, argparse, time
import threading
import queue
import webbrowser
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import dropbox, dropbox.files
from dropbox import DropboxOAuth2FlowNoRedirect

def get_token_file_path() -> Path:
  """Get the path to the token storage file."""
  config_dir = Path.home() / '.config' / 'dbox_upload'
  config_dir.mkdir(parents=True, exist_ok=True)
  return config_dir / 'token.json'

def save_token(token: str, refresh_token: Optional[str] = None, app_key: Optional[str] = None, app_secret: Optional[str] = None) -> None:
  """Save the access token and refresh token to a file."""
  token_file = get_token_file_path()
  token_data = {
    'access_token': token,
    'refresh_token': refresh_token,
    'app_key': app_key,
    'app_secret': app_secret,
    'timestamp': time.time()
  }
  with open(token_file, 'w') as f:
    json.dump(token_data, f)
  # Set restrictive permissions (readable/writable by owner only)
  os.chmod(token_file, 0o600)
  print(f"Token saved to {token_file}")

def load_saved_token_data() -> Optional[Dict[str, Any]]:
  """Load previously saved token data including refresh token."""
  token_file = get_token_file_path()
  if token_file.exists():
    try:
      with open(token_file, 'r') as f:
        return json.load(f)
    except (json.JSONDecodeError, KeyError):
      return None
  return None

def refresh_access_token(refresh_token: str, app_key: str, app_secret: str) -> Optional[str]:
  """Refresh the access token using a refresh token."""
  try:
    dbx = dropbox.Dropbox(
      oauth2_refresh_token=refresh_token,
      app_key=app_key,
      app_secret=app_secret
    )
    # This will automatically refresh the token
    dbx.users_get_current_account()
    # Get the new access token
    return dbx._oauth2_access_token
  except Exception as e:
    print(f"Failed to refresh token: {e}")
    return None

def clear_saved_token() -> None:
  """Clear the saved token."""
  token_file = get_token_file_path()
  if token_file.exists():
    token_file.unlink()
    print("Saved token cleared")

def get_auth_token_interactive() -> str:
  """Get Dropbox auth token through interactive OAuth flow."""
  print("\n" + "="*60)
  print("Dropbox Authentication Required")
  print("="*60)

  # Try to get app key and secret from environment variables first
  app_key = os.getenv('DBOX_KEY')
  app_secret = os.getenv('DBOX_SECRET')
  
  if app_key and app_secret:
    print("\nUsing Dropbox app credentials from DBOX_KEY and DBOX_SECRET environment variables")
  else:
    # You'll need to register your app at https://www.dropbox.com/developers/apps
    # For personal use, you can use your own app key and secret
    print("\nTo authenticate, you'll need a Dropbox app. If you don't have one:")
    print("1. Go to https://www.dropbox.com/developers/apps")
    print("2. Create a new app (choose 'Scoped access' and 'Full Dropbox')")
    print("3. Note your App key and App secret")
    print("\nTip: Set DBOX_KEY and DBOX_SECRET environment variables to skip this step")
    
    if not app_key:
      app_key = input("\nEnter your Dropbox App Key: ").strip()
    if not app_secret:
      app_secret = input("Enter your Dropbox App Secret: ").strip()
    
    if not app_key or not app_secret:
      raise ValueError("App key and secret are required")

  # Start OAuth flow with offline access for refresh tokens
  auth_flow = DropboxOAuth2FlowNoRedirect(
    app_key, 
    app_secret,
    token_access_type='offline'  # This requests refresh tokens
  )
  authorize_url = auth_flow.start()

  print("\n" + "="*60)
  print("Please visit this URL to authorize the application:")
  print(authorize_url)
  print("="*60)

  # Try to open the browser automatically
  try:
    webbrowser.open(authorize_url)
    print("\nThe authorization page should have opened in your browser.")
  except:
    print("\nCouldn't open browser automatically. Please copy and paste the URL above.")

  print("\nAfter authorizing, you'll get an authorization code.")
  auth_code = input("Enter the authorization code here: ").strip()

  if not auth_code:
    raise ValueError("Authorization code is required")

  try:
    oauth_result = auth_flow.finish(auth_code)
    access_token = oauth_result.access_token
    refresh_token = oauth_result.refresh_token if hasattr(oauth_result, 'refresh_token') else None

    if not refresh_token:
      print("\nWarning: No refresh token received. Long-running uploads may fail.")
      print("Make sure your Dropbox app has 'offline' access type configured.")
    
    # Test the token
    dbx = dropbox.Dropbox(access_token)
    account = dbx.users_get_current_account()
    print(f"\n✓ Successfully authenticated as: {account.name.display_name}")
    print(f"  Email: {account.email}")
    if refresh_token:
      print("  ✓ Refresh token obtained (supports long-running uploads)")

    # Ask if user wants to save the token
    save_choice = input("\nSave this token for future use? (y/n): ").strip().lower()
    if save_choice == 'y':
      save_token(access_token, refresh_token, app_key, app_secret)

    return {'access_token': access_token, 'refresh_token': refresh_token, 'app_key': app_key, 'app_secret': app_secret}

  except Exception as e:
    raise ValueError(f"Failed to complete authorization: {e}")

def get_or_request_token(args_token: Optional[str] = None) -> Dict[str, Any]:
  """Get token from args, saved file, or interactive prompt. Returns dict with token info."""
  # Priority: command line arg > saved token > interactive

  # 1. Command line argument (simple token only, no refresh)
  if args_token:
    print("Using token from command line argument")
    print("Warning: Command-line tokens don't support automatic renewal. Use 'auth' command for long uploads.")
    return {'access_token': args_token, 'refresh_token': None, 'app_key': None, 'app_secret': None}

  # 2. Saved token with refresh capability
  saved_data = load_saved_token_data()
  if saved_data:
    access_token = saved_data.get('access_token')
    refresh_token = saved_data.get('refresh_token')
    app_key = saved_data.get('app_key')
    app_secret = saved_data.get('app_secret')
    
    # If we have refresh token info, use it to get a fresh access token
    if refresh_token and app_key and app_secret:
      try:
        # Create Dropbox client with refresh token
        dbx = dropbox.Dropbox(
          oauth2_refresh_token=refresh_token,
          app_key=app_key,
          app_secret=app_secret
        )
        # Test and get fresh token
        dbx.users_get_current_account()
        fresh_token = dbx._oauth2_access_token
        print("✓ Using saved refresh token for automatic token renewal")
        return {'access_token': fresh_token, 'refresh_token': refresh_token, 'app_key': app_key, 'app_secret': app_secret}
      except Exception as e:
        print(f"Failed to use refresh token: {e}")
        clear_saved_token()
    elif access_token:
      # Try the saved access token
      try:
        dbx = dropbox.Dropbox(access_token)
        dbx.users_get_current_account()
        print("Using saved access token")
        return {'access_token': access_token, 'refresh_token': None, 'app_key': None, 'app_secret': None}
      except:
        print("Saved token is no longer valid")
        clear_saved_token()

  # 3. Interactive authentication
  print("\nNo valid token found. Starting authentication process...")
  return get_auth_token_interactive()

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

def get_dropbox_files(dbx: dropbox.Dropbox, folder_path: str) -> Set[str]:
  """Get set of file paths that exist in Dropbox folder."""
  existing_files = set()
  try:
    result = dbx.files_list_folder(folder_path, recursive=True)
    while True:
      for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
          existing_files.add(entry.path_lower)
      if not result.has_more:
        break
      result = dbx.files_list_folder_continue(result.cursor)
  except dropbox.exceptions.ApiError as e:
    if e.error.is_path() and e.error.get_path().is_not_found():
      # Folder doesn't exist yet, that's OK
      pass
    else:
      print(f"Error listing Dropbox folder: {e}")
  return existing_files

def get_local_files(local_path: str, extensions: tuple = ('.zip', '.7z', '.rar')) -> List[Dict[str, Any]]:
  """Get list of files to upload from local directory."""
  files = []

  if os.path.isfile(local_path):
    # Single file
    if local_path.endswith(extensions):
      files.append({
        'local_path': local_path,
        'relative_path': os.path.basename(local_path),
        'size': os.path.getsize(local_path)
      })
  elif os.path.isdir(local_path):
    # Directory - walk recursively
    for root, dirs, filenames in os.walk(local_path):
      for filename in filenames:
        if filename.endswith(extensions):
          file_path = os.path.join(root, filename)
          relative_path = os.path.relpath(file_path, local_path)
          files.append({
            'local_path': file_path,
            'relative_path': relative_path,
            'size': os.path.getsize(file_path)
          })

  return files


def upload_directory(
    token_info: Dict[str, Any],
    local_path: str,
    upload_folder: str,
    chunk_size_mb: int = 8,
    num_workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
    extensions: tuple = ('.zip', '.7z', '.rar')
):
  """Upload files from local directory to Dropbox folder."""
  # Create Dropbox client with refresh token if available
  if token_info['refresh_token'] and token_info['app_key'] and token_info['app_secret']:
    dbx = dropbox.Dropbox(
      oauth2_refresh_token=token_info['refresh_token'],
      app_key=token_info['app_key'],
      app_secret=token_info['app_secret']
    )
    print("Using refresh token for automatic token renewal during uploads")
  else:
    dbx = dropbox.Dropbox(token_info['access_token'])

  # Get local files
  print("Discovering local files...")
  local_files = get_local_files(local_path, extensions)

  if not local_files:
    print(f"No files with extensions {extensions} found to upload.")
    return

  print(f"Found {len(local_files)} local files")

  # Get existing Dropbox files
  print("Checking existing files on Dropbox...")
  existing_files = get_dropbox_files(dbx, upload_folder)
  print(f"Found {len(existing_files)} existing files on Dropbox")

  # Determine which files to upload
  files_to_upload = []
  total_size = 0

  for file_info in local_files:
    # Calculate destination path
    # Ensure upload_folder doesn't end with slash to avoid double slashes
    upload_folder_clean = upload_folder.rstrip('/')
    if os.path.isdir(local_path):
      # Preserve directory structure
      dest_path = upload_folder_clean + '/' + file_info['relative_path'].replace(os.sep, '/')
    else:
      # Single file
      dest_path = upload_folder_clean + '/' + os.path.basename(file_info['local_path'])

    # Check if file exists
    if dest_path.lower() not in existing_files or overwrite:
      files_to_upload.append({
        'local_path': file_info['local_path'],
        'dest_path': dest_path,
        'size': file_info['size']
      })
      total_size += file_info['size']

  print(f"\nFiles to upload: {len(files_to_upload)} of {len(local_files)}")
  print(f"Total size to upload: {format_bytes(total_size)}")

  if len(files_to_upload) == 0:
    print("All files already exist on Dropbox. Nothing to upload.")
    return

  if dry_run:
    print("\nDRY RUN MODE - No files will be uploaded")
    print("\nWould upload the following files:")
    for file_info in files_to_upload:
      print(f"  {file_info['local_path']} -> {file_info['dest_path']} ({format_bytes(file_info['size'])})")
    return

  # Upload files
  print("\nProceeding with upload...")
  print(f"Using {num_workers} worker threads with {chunk_size_mb}MB chunks")
  print("=" * 60)

  uploaded = 0
  failed = 0
  skipped = len(local_files) - len(files_to_upload)
  
  # Track overall progress
  total_bytes_uploaded = 0
  overall_start_time = time.time()

  for i, file_info in enumerate(files_to_upload, 1):
    print(f"\n[{i}/{len(files_to_upload)}] Uploading {os.path.basename(file_info['local_path'])} ({format_bytes(file_info['size'])})")
    
    # Calculate overall progress
    overall_elapsed = time.time() - overall_start_time
    if overall_elapsed > 0 and total_bytes_uploaded > 0:
      overall_rate = total_bytes_uploaded / overall_elapsed
      remaining_bytes = total_size - total_bytes_uploaded
      overall_eta = remaining_bytes / overall_rate if overall_rate > 0 else 0
      print(f"Overall: {format_bytes(total_bytes_uploaded)}/{format_bytes(total_size)} uploaded | Rate: {format_bytes(overall_rate)}/s | Total ETA: {format_time(overall_eta)}")

    # Progress bar for individual file
    progress_bars = 40  # Reduced to make room for more info
    print(f"File: [{' ' * progress_bars}] 0%", end='', flush=True)

    # Track progress for this file
    file_start_time = time.time()
    last_update_time = file_start_time
    file_bytes_uploaded = 0

    # Upload with progress tracking
    def progress_callback(uploaded_bytes, total_bytes):
      nonlocal last_update_time, file_bytes_uploaded
      file_bytes_uploaded = uploaded_bytes
      current_time = time.time()
      if current_time - last_update_time >= 0.5:  # Update every 0.5 seconds
        elapsed = current_time - file_start_time
        rate = uploaded_bytes / elapsed if elapsed > 0 else 0
        progress = (uploaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        filled = int((progress / 100) * progress_bars)

        remaining = total_bytes - uploaded_bytes
        eta = remaining / rate if rate > 0 else 0
        
        # Calculate overall ETA
        overall_elapsed = current_time - overall_start_time
        current_total = total_bytes_uploaded + uploaded_bytes
        if overall_elapsed > 0 and current_total > 0:
          overall_rate = current_total / overall_elapsed
          overall_remaining = total_size - current_total
          overall_eta = overall_remaining / overall_rate if overall_rate > 0 else 0
          overall_eta_str = f" | Total ETA: {format_time(overall_eta)}"
        else:
          overall_eta_str = ""

        print(f"\rFile: [{'█' * filled}{' ' * (progress_bars - filled)}] {progress:.1f}% | {format_bytes(rate)}/s | ETA: {format_time(eta)}{overall_eta_str}", end='', flush=True)
        last_update_time = current_time

    # Modified upload_file call with progress tracking
    success = upload_file_with_progress(
      dbx,
      file_info['local_path'],
      file_info['dest_path'],
      chunk_size_mb,
      num_workers,
      progress_callback
    )

    if success:
      uploaded += 1
      total_bytes_uploaded += file_info['size']
      elapsed = time.time() - file_start_time
      rate = file_info['size'] / elapsed if elapsed > 0 else 0
      print(f"\rFile: [{'█' * progress_bars}] 100% | {format_bytes(rate)}/s | Time: {format_time(elapsed)} | ✓ Complete")
    else:
      failed += 1
      print(f"\rFile: Failed to upload")
      # Still count partial upload in total bytes for better ETA estimation
      total_bytes_uploaded += file_bytes_uploaded

  print("\n" + "=" * 60)
  total_time = time.time() - overall_start_time
  average_rate = total_bytes_uploaded / total_time if total_time > 0 else 0
  
  print(f"Upload completed:")
  print(f"  Uploaded: {uploaded} files ({format_bytes(total_bytes_uploaded)})")
  print(f"  Skipped: {skipped}")
  print(f"  Failed: {failed}")
  print(f"  Total time: {format_time(total_time)}")
  print(f"  Average rate: {format_bytes(average_rate)}/s")

def upload_file_with_progress(
    dbx: dropbox.Dropbox,
    file_path: str,
    dest_path: str,
    chunk_size_mb: int = 8,
    num_workers: int = 4,
    progress_callback = None
):
  """Upload file with optional progress callback."""
  file_size = os.path.getsize(file_path)
  CHUNK_SIZE = chunk_size_mb * (2 ** 20)

  with open(file_path, 'rb') as f:
    # Small file - single upload
    if file_size <= CHUNK_SIZE:
      try:
        dbx.files_upload(f.read(), dest_path)
        if progress_callback:
          progress_callback(file_size, file_size)
        return True
      except Exception as e:
        print(f"\nError uploading {file_path}: {e}")
        return False

    # Large file - use existing chunked upload logic
    # (Reuse the existing concurrent upload code but with simpler output)
    try:
      upload_session_start_result = dbx.files_upload_session_start(
        b'',
        session_type=dropbox.files.UploadSessionType.concurrent,
      )

      chunk_queue = queue.Queue(maxsize=num_workers * 2)
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

      uploaded_size = 0
      while f.tell() < file_size:
        chunk_offset = f.tell()
        chunk_data = f.read(CHUNK_SIZE)
        uploaded_size += len(chunk_data)
        is_last = uploaded_size == file_size
        chunk_queue.put((chunk_data, chunk_offset, is_last))

        if progress_callback:
          progress_callback(uploaded_size, file_size)

      chunk_queue.join()
      stop_event.set()

      final_cursor = dropbox.files.UploadSessionCursor(
        session_id=upload_session_start_result.session_id,
        offset=uploaded_size,
      )
      commit = dropbox.files.CommitInfo(path=dest_path)
      dbx.files_upload_session_finish(b'', final_cursor, commit)

      return True

    except Exception as e:
      print(f"\nError uploading {file_path}: {e}")
      return False
    finally:
      stop_event.set()
      for worker in workers:
        worker.join(timeout=1.0)

def main():
  parser = argparse.ArgumentParser(
    description='Upload files or directories to Dropbox',
    epilog='Authentication: The script will use token in this order: --token arg, saved token (with refresh support), or interactive OAuth. Run "auth" command first for best experience.'
  )

  # Add subcommands for token management
  subparsers = parser.add_subparsers(dest='command', help='Commands', required=True)

  # Upload command
  upload_parser = subparsers.add_parser('upload', help='Upload files to Dropbox')
  upload_parser.add_argument('local_path', type=str, help='Path to file or directory to upload')
  upload_parser.add_argument('upload_path', type=str, help='Destination path in Dropbox')
  upload_parser.add_argument('--token', type=str, help='Dropbox access token (optional)')
  upload_parser.add_argument('--chunk_size_mb', type=int, default=8, help='Chunk size in MB (default: 8)')
  upload_parser.add_argument('--workers', type=int, default=4, help='Number of worker threads for concurrent uploads (default: 4)')
  upload_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
  upload_parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without actually uploading')
  upload_parser.add_argument('--extensions', type=str, default='.zip,.7z,.rar', help='Comma-separated list of file extensions to upload (default: .zip,.7z,.rar)')

  # Auth command - authenticate and save token
  auth_parser = subparsers.add_parser('auth', help='Authenticate and save token for future use')

  # Clear command - clear saved token
  clear_parser = subparsers.add_parser('clear-token', help='Clear saved authentication token')

  args = parser.parse_args()

  # Handle commands
  if args.command == 'auth':
    # Just authenticate and save token
    try:
      token_info = get_auth_token_interactive()
      print("\n✓ Authentication successful!")
    except Exception as e:
      print(f"\n✗ Authentication failed: {e}")
      return

  elif args.command == 'clear-token':
    # Clear saved token
    clear_saved_token()

  elif args.command == 'upload':
    # Get token using the new system
    try:
      token_info = get_or_request_token(args.token)
    except Exception as e:
      print(f"Failed to get authentication token: {e}")
      return

    # Parse extensions
    extensions = tuple(ext.strip() for ext in args.extensions.split(','))

    # Check if path exists
    if not os.path.exists(args.local_path):
      print(f"Error: {args.local_path} does not exist")
      return

    upload_directory(
        token_info=token_info,
        local_path=args.local_path,
        upload_folder=args.upload_path,
        chunk_size_mb=args.chunk_size_mb,
        num_workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        extensions=extensions
    )

if __name__ == "__main__":
  main()