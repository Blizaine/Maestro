"""Optional covers install only notation packages and remain cancellable."""
import subprocess
import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('score_dependencies', Path(__file__).resolve().parents[1] / 'app/models/TTS/yue2/score_dependencies.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class ScoreDependenciesTests(unittest.TestCase):
    def test_existing_runtime_needs_no_installer(self):
        with patch.object(score.importlib.util, 'find_spec', return_value=object()), \
                patch.object(score.subprocess, 'Popen') as launch:
            score.ensure_score_dependencies()
            launch.assert_not_called()

    def test_missing_transitive_dependency_is_installed_without_resolving_core_stack(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ('Installed', None)
        process.poll.return_value = 0
        with patch.object(score.importlib.util, 'find_spec', side_effect=lambda name: None if name == 'importlib_resources' else object()), \
                patch.object(score.shutil, 'which', return_value=None), \
                patch.object(score.subprocess, 'Popen', return_value=process) as launch:
            score.ensure_score_dependencies()
        args = launch.call_args.args[0]
        self.assertEqual(args[:4], [sys.executable, '-m', 'pip', 'install'])
        self.assertIn('--no-deps', args)
        self.assertEqual(args[-1], 'importlib_resources==6.5.2')
        self.assertFalse(any('torch' in argument or argument == '--upgrade' for argument in args))

    def test_cancelled_download_terminates_installer_before_releasing_job(self):
        process = Mock(returncode=None)
        process.communicate.side_effect = [subprocess.TimeoutExpired('uv', 0.25), ('cancelled', None)]
        process.poll.return_value = None
        with patch.object(score.importlib.util, 'find_spec', return_value=None), \
                patch.object(score.shutil, 'which', return_value='uv'), \
                patch.object(score.subprocess, 'Popen', return_value=process):
            with self.assertRaises(InterruptedError):
                score.ensure_score_dependencies(Mock(side_effect=[False, False, True]))
        process.terminate.assert_called_once()
        process.communicate.assert_called_with(timeout=5)


if __name__ == '__main__':
    unittest.main()
