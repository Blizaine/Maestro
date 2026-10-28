from __future__ import annotations

import copy

import pytest

from app.services.editor_operations import (
    EditorOperationError,
    attach_job,
    create_operation,
    find_owned_job,
    get_operation,
    reconcile_operation,
    source_snapshot_matches,
    transition_operation,
    validate_submission,
)


def _payload(operation_id: str = "editor-op-0001", submission_id: str = "submission-0001") -> dict:
    return {
        "operation_id": operation_id,
        "client_submission_id": submission_id,
        "workspace": "workspace-a",
        "project_id": "project-a",
        "project_revision": 7,
        "item_id": "clip-a",
        "item_name": "Opening shot",
        "tool": "retake",
        "return_mode": "replace",
        "source_snapshot": {
            "item_id": "clip-a",
            "track_id": "video-1",
            "track_type": "video",
            "asset_id": "asset-a",
            "asset_name": "source.mp4",
            "asset_path": "C:/workspace/source.mp4",
            "start": 2.0,
            "duration": 4.0,
            "source_in": 1.0,
            "speed": 1.0,
        },
        "source_range": {"start": 1.0, "end": 5.0},
    }


def test_interleaved_jobs_reconcile_only_the_exact_owned_job(tmp_path):
    operation = create_operation(tmp_path, _payload())
    attach_job(tmp_path, operation["operation_id"], operation["client_submission_id"], "job-editor-1", "retake")

    unrelated = {
        "id": "job-studio-2",
        "status": "completed",
        "client_submission_id": "submission-other",
        "editor_operation_id": None,
        "output_files": ["unrelated.mp4"],
    }
    owned = {
        "id": "job-editor-1",
        "status": "completed",
        "client_submission_id": operation["client_submission_id"],
        "editor_operation_id": operation["operation_id"],
        "output_files": ["editor-result.mp4"],
    }

    reconciled = reconcile_operation(
        tmp_path,
        operation["operation_id"],
        {unrelated["id"]: unrelated, owned["id"]: owned},
    )

    assert find_owned_job(reconciled, [unrelated, owned]) == owned
    assert reconciled["job_id"] == owned["id"]
    assert reconciled["status"] == "result_ready"
    assert reconciled["output_file"] == "editor-result.mp4"


def test_operation_survives_reload_and_never_claims_a_new_job(tmp_path):
    operation = create_operation(tmp_path, _payload())
    reloaded = get_operation(tmp_path, operation["operation_id"])
    assert reloaded["status"] == "armed"
    assert reloaded["job_id"] is None

    unrelated = {
        "id": "job-unrelated",
        "status": "running",
        "client_submission_id": "submission-other",
        "editor_operation_id": None,
        "output_files": [],
    }
    assert find_owned_job(reloaded, [unrelated]) is None
    still_armed = reconcile_operation(tmp_path, operation["operation_id"], {unrelated["id"]: unrelated})
    assert still_armed["status"] == "armed"
    assert still_armed["job_id"] is None


def test_terminal_transition_requires_exact_operation_and_job_ownership(tmp_path):
    operation = create_operation(tmp_path, _payload())
    attach_job(tmp_path, operation["operation_id"], operation["client_submission_id"], "job-editor-1", "retake")
    reconcile_operation(tmp_path, operation["operation_id"], {
        "job-editor-1": {
            "id": "job-editor-1",
            "status": "completed",
            "client_submission_id": operation["client_submission_id"],
            "editor_operation_id": operation["operation_id"],
            "output_files": ["result.mp4"],
        },
    })

    with pytest.raises(EditorOperationError, match="does not own"):
        transition_operation(
            tmp_path,
            operation["operation_id"],
            operation["client_submission_id"],
            "completed",
            job_id="job-foreign-1",
            output_file="result.mp4",
        )

    completed = transition_operation(
        tmp_path,
        operation["operation_id"],
        operation["client_submission_id"],
        "completed",
        job_id="job-editor-1",
        output_file="result.mp4",
    )
    assert completed["status"] == "completed"


def test_source_snapshot_detects_replaced_deleted_or_retimed_clip_but_allows_other_timeline_edits():
    before = _payload()["source_snapshot"]
    after_unrelated_edit = copy.deepcopy(before)
    after_unrelated_edit["start"] = 17.0
    after_unrelated_edit["track_id"] = "video-2"
    assert source_snapshot_matches(before, after_unrelated_edit)

    retimed = copy.deepcopy(before)
    retimed["source_in"] = 1.5
    assert not source_snapshot_matches(before, retimed)

    replaced = copy.deepcopy(before)
    replaced["asset_id"] = "asset-b"
    assert not source_snapshot_matches(before, replaced)

    deleted = None
    assert not source_snapshot_matches(before, deleted)


def test_operation_file_path_is_confined_to_storage_root(tmp_path):
    operation = create_operation(tmp_path, _payload())
    assert get_operation(tmp_path, operation["operation_id"])["operation_id"] == operation["operation_id"]
    with pytest.raises(EditorOperationError, match="Invalid operation id"):
        get_operation(tmp_path, "../outside")


def test_duplicate_create_is_idempotent_only_for_same_identity(tmp_path):
    first = create_operation(tmp_path, _payload())
    second = create_operation(tmp_path, _payload())
    assert second["operation_id"] == first["operation_id"]
    with pytest.raises(EditorOperationError, match="already exists"):
        create_operation(tmp_path, _payload(submission_id="submission-different"))


def test_operation_rejects_helper_or_wrong_tool_submission_before_job_link(tmp_path):
    operation = create_operation(tmp_path, _payload())
    with pytest.raises(EditorOperationError, match="requested tool"):
        validate_submission(
            tmp_path,
            operation["operation_id"],
            operation["client_submission_id"],
            expected_tool="recast",
        )
    with pytest.raises(EditorOperationError, match="job kind"):
        attach_job(
            tmp_path,
            operation["operation_id"],
            operation["client_submission_id"],
            "job-helper-1",
            "generation",
        )
    still_armed = get_operation(tmp_path, operation["operation_id"])
    assert still_armed["status"] == "armed"
    assert still_armed["job_id"] is None
