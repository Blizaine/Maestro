"""Verify the bundled native allocator companions against their build evidence."""
import hashlib
import json
from pathlib import Path
import struct

DIRECTORY = Path(__file__).resolve().parents[1] / 'app/mmgp_backends/v43/mmgp/allocator'

def test_oom_companion_is_exact_pinned_source_build_with_windows_x64_pe():
    proof = json.loads((DIRECTORY / 'OOM_BUILD.json').read_text())
    binary = (DIRECTORY / proof['artifact']).read_bytes()
    # Vendored bytes are intentionally -text in .gitattributes, including their
    # upstream line endings; the build proof binds those exact source bytes.
    source = (DIRECTORY / proof['source_file']).read_bytes()
    assert proof['upstream_commit'] == '9905f0df941e5ec7b0c80e32555d3a61fe415726'
    assert len(binary) == proof['artifact_bytes']
    assert hashlib.sha256(binary).hexdigest() == proof['artifact_sha256']
    assert hashlib.sha256(source).hexdigest() == proof['source_sha256']
    assert binary[:2] == b'MZ'
    offset = struct.unpack_from('<I', binary, 0x3C)[0]
    assert binary[offset:offset+4] == b'PE\0\0'
    assert struct.unpack_from('<H', binary, offset + 4)[0] == 0x8664
    assert b'mmgp_throw_out_of_memory\0' in binary
    assert b'c10.dll\0' in binary

def test_oom_companion_evidence_keeps_gpu_validation_boundary_explicit():
    proof = json.loads((DIRECTORY / 'OOM_BUILD.json').read_text())
    checks = proof['cpu_exception_checks']
    assert {row['torch'].split('+')[0] for row in checks} == {'2.7.1', '2.10.0'}
    assert all(row['companion_load'] and row['cross_dll_c10_out_of_memory_type_and_message'] for row in checks)
    assert all(row['cuda_initialized'] is False for row in checks)
    assert 'not a real GPU OOM/recovery certification' in proof['validation_note']
