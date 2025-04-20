import os
import shutil
import tempfile
import unittest
import zipfile

from slippi_db import utils

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

if __name__ == '__main__':
    unittest.main()
