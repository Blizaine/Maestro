"""Read-only receipt recovery covers terminal jobs omitted by the active list."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


class RouteError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class EditorTitleReceiptTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'app' / 'launch.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'job_by_submission')
        node.decorator_list = []
        self.jobs = {
            'done': {'id': 'done', 'client_submission_id': 'exact-title-id', 'status': 'completed'},
            'other': {'id': 'other', 'client_submission_id': 'unrelated', 'status': 'running'},
        }
        self.reads = []

        def status(job_id):
            self.reads.append(job_id)
            return {**self.jobs[job_id], 'job_id': job_id}

        namespace = {'HTTPException': RouteError, '_jobs': self.jobs, 'get_status': status}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
        self.lookup = namespace['job_by_submission']

    def test_completed_and_failed_receipts_remain_recoverable_without_submission(self):
        for terminal in ('completed', 'failed', 'cancelled'):
            self.jobs['done']['status'] = terminal
            self.assertEqual(self.lookup('exact-title-id')['status'], terminal)
        self.assertEqual(self.reads, ['done', 'done', 'done'])
        self.assertEqual(len(self.jobs), 2)

    def test_exact_identity_is_required_and_missing_receipt_is_404(self):
        with self.assertRaises(RouteError) as raised:
            self.lookup('exact-title')
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(self.reads, [])

    def test_ambiguous_identity_is_rejected_without_choosing_a_job(self):
        self.jobs['collision'] = {'id': 'collision', 'client_submission_id': 'exact-title-id', 'status': 'queued'}
        with self.assertRaises(RouteError) as raised:
            self.lookup('exact-title-id')
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.reads, [])

    def test_unbounded_identity_is_rejected_before_scanning(self):
        with self.assertRaises(RouteError) as raised:
            self.lookup('x' * 161)
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == '__main__':
    unittest.main()
