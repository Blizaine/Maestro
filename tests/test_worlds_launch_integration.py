"""Exercise the real API adapters without importing the GPU application."""
import ast
import asyncio
from pathlib import Path

from fastapi import HTTPException
import pytest


def adapters(prepare, jobs=None):
    path = Path(__file__).parents[1] / 'app/launch.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_worlds_submit_generation', '_worlds_job_status'}
    nodes = [node for node in tree.body if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in names]
    scope = {'_prepare_generation_submission': prepare, '_jobs': jobs or {},
             'HTTPException': HTTPException, 'snapshot_job': lambda job: dict(job),
             'get_status': lambda job_id: {'job_id': job_id, 'status': 'completed'}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def test_worlds_uses_regular_submission_with_fixed_workspace_and_client_owner():
    seen = []
    async def prepare(body):
        seen.append(body)
        return {'job_id': 'owned', 'client_submission_id': body['_client_submission_id']}
    source = {'prompt': 'world', 'model_type': 'hy_nomas', '_client_submission_id': 'uuid',
              'custom_settings': {'world_output': 'world_video', 'acknowledge_pixelumm_noncommercial': True}}
    result = asyncio.run(adapters(prepare)['_worlds_submit_generation'](source))
    assert result['job_id'] == 'owned'
    assert seen[0]['workspace'] == 'Worlds'
    assert seen[0]['_client_submission_id'] == 'uuid'
    assert seen[0]['custom_settings'] == source['custom_settings']
    assert seen[0]['_enhance_on_generation'] is False
    assert 'workspace' not in source


def test_preflight_rejection_is_terminal_only_without_an_accepted_job():
    async def rejected(_):
        raise HTTPException(400, 'Invalid saved world')
    scope = adapters(rejected)
    result = asyncio.run(scope['_worlds_submit_generation']({'_client_submission_id': 'uuid'}))
    assert result == {'rejected': True, 'client_submission_id': 'uuid', 'detail': 'Invalid saved world'}
    jobs = {'owned': {'id': 'owned', 'status': 'running', 'client_submission_id': 'uuid'}}
    result = asyncio.run(adapters(rejected, jobs)['_worlds_submit_generation']({'_client_submission_id': 'uuid'}))
    assert result['job_id'] == 'owned' and 'rejected' not in result
    async def uncertain(_):
        raise HTTPException(503, 'Temporary failure')
    with pytest.raises(HTTPException):
        asyncio.run(adapters(uncertain)['_worlds_submit_generation']({'_client_submission_id': 'uuid'}))


def test_private_status_passes_frozen_directory_without_changing_public_api():
    jobs = {'owned': {'id': 'owned', 'out_dir': 'fixed/server/path', 'params': {'seed': 42}}}
    status = adapters(None, jobs)['_worlds_job_status']('owned')
    assert status['out_dir'] == 'fixed/server/path'
    assert status['params'] == {'seed': 42}
