"""Read-only receipt recovery covers terminal jobs omitted by the active list."""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import sys

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
from services.agent_access.auth import Principal
from services.agent_access.backend import AgentAccessBackend
from services.agent_access.context import principal_context
from services.generation_details import public_generation_details
from services.job_lifecycle import snapshot_job


class RouteError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class EditorTitleReceiptTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'app' / 'launch.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'job_by_submission', '_agent_require_job_access', '_director_job_response_fields', 'get_status'}
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                node.decorator_list = []
                selected.append(node)

        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        app_root = root / 'app'
        app_root.mkdir()
        backend = AgentAccessBackend(
            SimpleNamespace(directory=root / 'settings', backend=lambda: {}),
            app_root,
            recover_previous=False,
        )
        self.jobs = backend.registry()
        self.actor_a = Principal('actor-a', 'client-a', ('observe',))
        self.actor_b = Principal('actor-b', 'client-b', ('observe',))
        base = {
            'status': 'completed', 'progress': 100, 'step': 0,
            'total_steps': 0, 'phase': '', 'message': 'Done',
            'output_files': [], 'error': None, 'params': {},
        }
        for identifier, submission, actor in (
            ('done', 'exact-title-id', self.actor_a),
            ('other', 'unrelated', self.actor_b),
        ):
            token = principal_context.set(actor)
            try:
                self.jobs[identifier] = {
                    **base, 'id': identifier,
                    'client_submission_id': submission,
                }
            finally:
                principal_context.reset(token)
        self.reads = []

        namespace = {
            'HTTPException': RouteError,
            '_jobs': self.jobs,
            '_agent_access_backend': backend,
            'principal_context': principal_context,
            'snapshot_job': lambda job: self._snapshot(job),
            'public_generation_details': public_generation_details,
            'public_enhancement': lambda value, **_kwargs: value,
            '_director_job_response_fields': lambda _job: {},
            '_generation_previews': SimpleNamespace(fields=lambda _job: {}),
            '_job_eta_response_fields': lambda _job: {},
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), namespace)
        self.lookup = namespace['job_by_submission']

    def _snapshot(self, job):
        self.reads.append(job['id'])
        return snapshot_job(job)

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

    def test_restricted_client_cannot_recover_another_owners_receipt(self):
        token = principal_context.set(self.actor_a)
        try:
            self.assertEqual(self.lookup('exact-title-id')['job_id'], 'done')
            with self.assertRaises(RouteError) as raised:
                self.lookup('unrelated')
        finally:
            principal_context.reset(token)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(self.reads, ['done'])


if __name__ == '__main__':
    unittest.main()
