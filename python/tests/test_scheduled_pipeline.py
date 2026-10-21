"""Publication failures; run explicitly inside the bounded scientific container."""
import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from mlrescurix_worker.scientific.scheduled_pipeline import run_pipeline

class PipelinePublicationTests(unittest.TestCase):
    def test_disk_full_copy_removes_partial_publication(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'trials/candidate'
            def partial(source,stage):
                Path(stage).mkdir()
                (Path(stage)/'partial.joblib').write_bytes(b'partial')
                raise OSError(errno.ENOSPC,'injected bounded publication failure')
            with patch('mlrescurix_worker.scientific.scheduled_pipeline._execute_pipeline',return_value={'status':'fixture'}),patch('mlrescurix_worker.scientific.scheduled_pipeline.shutil.copytree',side_effect=partial):
                with self.assertRaises(OSError):
                    run_pipeline(root,{'experiment_id':'candidate'})
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name('.candidate-staging').exists())

    def test_real_container_tmpfs_enospc_never_publishes(self):
        # Fail explicitly unless the test runner supplies a bounded tmpfs.
        import os
        if os.environ.get('MLRESCURIX_TEST_BOUNDED_TMPFS') != '64m':
            self.skipTest("Requires container execution with 64m tmpfs mount (MLRESCURIX_TEST_BOUNDED_TMPFS=64m)")
        with tempfile.TemporaryDirectory() as root:
            def fill(output,protocol,directory):
                with (directory/'bounded-fill').open('wb') as file:
                    for _ in range(65):
                        file.write(b'x'*(1024*1024))
                raise AssertionError('64 MiB tmpfs bound was not enforced')
            with patch('mlrescurix_worker.scientific.scheduled_pipeline._execute_pipeline',side_effect=fill):
                with self.assertRaises(OSError) as outcome:
                    run_pipeline(root,{'experiment_id':'candidate'})
            self.assertEqual(outcome.exception.errno,errno.ENOSPC)
            self.assertFalse((Path(root)/'trials/candidate').exists())

if __name__=='__main__':unittest.main()
