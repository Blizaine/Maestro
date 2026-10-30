"""CPU-only tests for the legacy MMGP pinned staging allocation guard.

The test compiles only the production ``_PinnedStager`` class into a fake
namespace, so no Torch import, CUDA initialization, or device API call occurs.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
OFFLOAD_SOURCE = ROOT / "app" / "mmgp" / "offload.py"
GIB = 2**30
STAGING_CHUNK_SIZE = 2**24


class _FakeSlot:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size


class _FakeEvent:
    def __init__(self, calls: list[tuple]):
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append(("event_synchronize",))


class PinnedStagerAllocationGuardTests(unittest.TestCase):
    def make_stager(
        self,
        *,
        available: int,
        physical: int = 32 * GIB,
        slots: tuple[_FakeSlot | None, _FakeSlot | None] = (None, None),
        events: tuple[_FakeEvent | None, _FakeEvent | None] = (None, None),
        allocation_error: RuntimeError | None = None,
        synchronize_error: RuntimeError | None = None,
    ):
        calls: list[tuple] = []

        class FakeCuda:
            def synchronize(self, device):
                calls.append(("cuda_synchronize", device))
                if synchronize_error is not None:
                    raise synchronize_error

        class FakeTorch:
            uint8 = object()
            cuda = FakeCuda()

            @staticmethod
            def inference_mode(enabled):
                return nullcontext()

            @staticmethod
            def empty(size, **kwargs):
                calls.append(("empty", size, kwargs))
                if allocation_error is not None:
                    raise allocation_error
                return _FakeSlot(size)

        namespace = {
            "STAGING_CHUNK_SIZE": STAGING_CHUNK_SIZE,
            "physical_memory": physical,
            "psutil": SimpleNamespace(
                virtual_memory=lambda: SimpleNamespace(available=available)
            ),
            "torch": FakeTorch,
        }
        class_node = next(
            node
            for node in ast.parse(OFFLOAD_SOURCE.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == "_PinnedStager"
        )
        exec(
            compile(
                ast.Module(body=[class_node], type_ignores=[]),
                str(OFFLOAD_SOURCE),
                "exec",
            ),
            namespace,
        )
        stager_type = namespace["_PinnedStager"]
        stager = stager_type.__new__(stager_type)
        stager.slots = list(slots)
        stager.events = list(events)
        stager.slot_no = 0
        stager.enabled = True
        stager.helpers = None
        return stager, calls

    def test_low_available_ram_skips_new_slot_and_preserves_both_slots(self):
        old_slot = _FakeSlot(1)
        other_slot = _FakeSlot(STAGING_CHUNK_SIZE)
        floor = 4 * GIB
        stager, calls = self.make_stager(
            available=floor + STAGING_CHUNK_SIZE - 1,
            slots=(other_slot, old_slot),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = stager._acquire(2, destination_device="cuda:1")

        self.assertIsNone(result)
        self.assertFalse(stager.enabled)
        self.assertIs(stager.slots[0], other_slot)
        self.assertIs(stager.slots[1], old_slot)
        self.assertEqual(calls, [])
        self.assertIn("Skipping Reserved RAM", output.getvalue())
        self.assertIn("4.00 GiB host reserve", output.getvalue())
        self.assertIn("pageable weights", output.getvalue())

    def test_reserve_is_ten_percent_of_physical_ram_when_larger(self):
        floor = 10 * GIB
        stager, calls = self.make_stager(
            available=floor + STAGING_CHUNK_SIZE - 1,
            physical=100 * GIB,
        )

        self.assertIsNone(stager._acquire(1))

        self.assertFalse(stager.enabled)
        self.assertEqual(calls, [])

    def test_new_slot_syncs_exact_destination_before_pinned_allocation(self):
        floor = 10 * GIB
        stager, calls = self.make_stager(
            available=floor + STAGING_CHUNK_SIZE,
            physical=100 * GIB,
        )

        slot = stager._acquire(1, destination_device="cuda:2")

        self.assertIsInstance(slot, _FakeSlot)
        self.assertEqual(slot.numel(), STAGING_CHUNK_SIZE)
        self.assertEqual(calls[0], ("cuda_synchronize", "cuda:2"))
        self.assertEqual(calls[1][0], "empty")
        self.assertEqual(calls[1][1], STAGING_CHUNK_SIZE)
        self.assertTrue(calls[1][2]["pin_memory"])
        self.assertEqual(calls[1][2]["device"], "cpu")

    def test_generic_call_without_cuda_destination_does_not_synchronize(self):
        stager, calls = self.make_stager(available=8 * GIB)

        slot = stager._acquire(1)

        self.assertIsInstance(slot, _FakeSlot)
        self.assertEqual([call[0] for call in calls], ["empty"])

    def test_adequate_existing_slot_keeps_event_reuse_behavior_under_pressure(self):
        slot = _FakeSlot(STAGING_CHUNK_SIZE)
        event = _FakeEvent([])
        stager, calls = self.make_stager(
            available=1,
            slots=(None, slot),
            events=(None, event),
        )

        result = stager._acquire(STAGING_CHUNK_SIZE // 2, "cuda:0")

        self.assertIs(result, slot)
        self.assertIsNone(stager.events[1])
        self.assertEqual(calls, [])
        self.assertEqual(event.calls, [("event_synchronize",)])
        self.assertTrue(stager.enabled)

    def test_sync_error_propagates_outside_pinned_allocation_fallback(self):
        previous_cuda_error = RuntimeError("prior asynchronous CUDA failure")
        stager, calls = self.make_stager(
            available=8 * GIB,
            synchronize_error=previous_cuda_error,
        )

        with self.assertRaisesRegex(RuntimeError, "prior asynchronous CUDA failure"):
            stager._acquire(1, destination_device="cuda:3")

        self.assertTrue(stager.enabled)
        self.assertEqual([call[0] for call in calls], ["cuda_synchronize"])

    def test_pin_allocation_error_is_reported_and_old_slot_is_retained(self):
        old_slot = _FakeSlot(1)
        pin_error = RuntimeError("simulated pinned allocation failure")
        stager, calls = self.make_stager(
            available=8 * GIB,
            slots=(None, old_slot),
            allocation_error=pin_error,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = stager._acquire(2, destination_device="cuda:4")

        self.assertIsNone(result)
        self.assertFalse(stager.enabled)
        self.assertIs(stager.slots[1], old_slot)
        self.assertEqual([call[0] for call in calls], ["cuda_synchronize", "empty"])
        self.assertIn("RuntimeError: simulated pinned allocation failure", output.getvalue())


if __name__ == "__main__":
    unittest.main()
