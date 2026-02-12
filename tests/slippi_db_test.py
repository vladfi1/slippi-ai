import os
import pickle
import shutil
import tempfile
import unittest
import zipfile

from slippi_db import utils
from slippi_db.utils import is_remote, FsspecFile

class CopyZipFilesTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.source_zip = os.path.join(self.test_dir, 'source.zip')
        with zipfile.ZipFile(self.source_zip, 'w') as zf:
            zf.writestr('file1.txt', 'This is file 1')
            zf.writestr('file2.txt', 'This is file 2')
            zf.writestr('file3.txt', 'This is file 3')
            zf.writestr('subdir/file4.txt', 'This is file 4 in a subdirectory')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_copy_to_new_destination(self):
        """Test copying files to a new destination archive."""
        dest_zip = os.path.join(self.test_dir, 'new_dest.zip')
        files_to_copy = ['file1.txt', 'file3.txt']

        utils.copy_zip_files(self.source_zip, files_to_copy, dest_zip)

        self.assertTrue(os.path.exists(dest_zip))
        with zipfile.ZipFile(dest_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 2)
            self.assertIn('file1.txt', file_list)
            self.assertIn('file3.txt', file_list)

            self.assertEqual(zf.read('file1.txt').decode('utf-8'), 'This is file 1')
            self.assertEqual(zf.read('file3.txt').decode('utf-8'), 'This is file 3')

    def test_copy_to_existing_destination(self):
        """Test copying files to an existing destination archive."""
        dest_zip = os.path.join(self.test_dir, 'existing_dest.zip')
        with zipfile.ZipFile(dest_zip, 'w') as zf:
            zf.writestr('existing_file.txt', 'This is an existing file')
            zf.writestr('file2.txt', 'This is an existing version of file2')

        files_to_copy = ['file1.txt', 'file2.txt']

        utils.copy_zip_files(self.source_zip, files_to_copy, dest_zip)

        with zipfile.ZipFile(dest_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 3)
            self.assertIn('existing_file.txt', file_list)
            self.assertIn('file1.txt', file_list)
            self.assertIn('file2.txt', file_list)

            self.assertEqual(zf.read('existing_file.txt').decode('utf-8'), 'This is an existing file')
            self.assertEqual(zf.read('file1.txt').decode('utf-8'), 'This is file 1')
            self.assertEqual(zf.read('file2.txt').decode('utf-8'), 'This is file 2')

    def test_copy_subdirectory_files(self):
        """Test copying files from subdirectories."""
        dest_zip = os.path.join(self.test_dir, 'subdir_dest.zip')
        files_to_copy = ['subdir/file4.txt']

        utils.copy_zip_files(self.source_zip, files_to_copy, dest_zip)

        with zipfile.ZipFile(dest_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 1)
            self.assertIn('subdir/file4.txt', file_list)

            self.assertEqual(zf.read('subdir/file4.txt').decode('utf-8'), 'This is file 4 in a subdirectory')


class DeleteFromZipTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.test_zip = os.path.join(self.test_dir, 'test.zip')
        with zipfile.ZipFile(self.test_zip, 'w') as zf:
            zf.writestr('file1.txt', 'This is file 1')
            zf.writestr('file2.txt', 'This is file 2')
            zf.writestr('file3.txt', 'This is file 3')
            zf.writestr('subdir/file4.txt', 'This is file 4 in a subdirectory')
            zf.writestr('subdir/file5.txt', 'This is file 5 in a subdirectory')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_delete_single_file(self):
        """Test deleting a single file from zip archive."""
        utils.delete_from_zip(self.test_zip, ['file2.txt'])

        with zipfile.ZipFile(self.test_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 4)
            self.assertNotIn('file2.txt', file_list)
            self.assertIn('file1.txt', file_list)
            self.assertIn('file3.txt', file_list)
            self.assertIn('subdir/file4.txt', file_list)
            self.assertIn('subdir/file5.txt', file_list)

    def test_delete_multiple_files(self):
        """Test deleting multiple files from zip archive."""
        utils.delete_from_zip(self.test_zip, ['file1.txt', 'file3.txt'])

        with zipfile.ZipFile(self.test_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 3)
            self.assertNotIn('file1.txt', file_list)
            self.assertNotIn('file3.txt', file_list)
            self.assertIn('file2.txt', file_list)
            self.assertIn('subdir/file4.txt', file_list)
            self.assertIn('subdir/file5.txt', file_list)

    def test_delete_subdirectory_file(self):
        """Test deleting a file from a subdirectory."""
        utils.delete_from_zip(self.test_zip, ['subdir/file4.txt'])

        with zipfile.ZipFile(self.test_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 4)
            self.assertNotIn('subdir/file4.txt', file_list)
            self.assertIn('subdir/file5.txt', file_list)

    def test_delete_empty_list(self):
        """Test deleting with an empty file list."""
        # Get original contents
        with zipfile.ZipFile(self.test_zip, 'r') as zf:
            original_files = zf.namelist()

        utils.delete_from_zip(self.test_zip, [])

        with zipfile.ZipFile(self.test_zip, 'r') as zf:
            file_list = zf.namelist()
            self.assertEqual(file_list, original_files)

    def test_delete_from_nonexistent_zip(self):
        """Test deleting from a non-existent zip file."""
        nonexistent_zip = os.path.join(self.test_dir, 'nonexistent.zip')

        with self.assertRaises(FileNotFoundError):
            utils.delete_from_zip(nonexistent_zip, ['file1.txt'])


class IsRemoteTest(unittest.TestCase):

    def test_s3_is_remote(self):
        self.assertTrue(is_remote('s3://bucket/path'))

    def test_gs_is_remote(self):
        self.assertTrue(is_remote('gs://bucket/path'))

    def test_http_is_remote(self):
        self.assertTrue(is_remote('https://example.com/data'))

    def test_local_path_not_remote(self):
        self.assertFalse(is_remote('/data/my-dataset'))

    def test_relative_path_not_remote(self):
        self.assertFalse(is_remote('data/my-dataset'))

    def test_file_uri_not_remote(self):
        self.assertFalse(is_remote('file:///data/my-dataset'))


class FsspecFileTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test.bin')
        self.test_content = b'hello fsspec'
        with open(self.test_file, 'wb') as f:
            f.write(self.test_content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fsspec_file_local_read(self):
        ff = FsspecFile(self.test_file)
        self.assertEqual(ff.read(), self.test_content)

    def test_fsspec_file_name(self):
        ff = FsspecFile('s3://bucket/games/abc123')
        self.assertEqual(ff.name, 'abc123')

    def test_fsspec_file_pickling(self):
        ff = FsspecFile('s3://bucket/games/abc123')
        ff2 = pickle.loads(pickle.dumps(ff))
        self.assertEqual(ff2.uri, ff.uri)

    def test_fsspec_file_extract(self):
        ff = FsspecFile(self.test_file)
        extract_dir = tempfile.mkdtemp()
        try:
            with ff.extract(extract_dir) as path:
                with open(path, 'rb') as f:
                    self.assertEqual(f.read(), self.test_content)
            self.assertFalse(os.path.exists(path))
        finally:
            shutil.rmtree(extract_dir)


if __name__ == '__main__':
    unittest.main()
