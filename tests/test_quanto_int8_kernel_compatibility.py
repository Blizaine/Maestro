"""CPU-only checks for the large-M Quanto Triton fallback path."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "app" / "shared" / "kernels"


def _extract(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class _FakeOutput:
    def __init__(self, shape):
        self.shape = shape

    def stride(self, dim):
        return (self.shape[1], 1)[dim]


class _FakeDevice:
    type = "cuda"
    index = 0


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.device = _FakeDevice()

    def stride(self, dim):
        return self.shape[1] if dim == 0 else 1


class _FakeTriton:
    @staticmethod
    def cdiv(size, block):
        return (size + block - 1) // block


class _FakeOutOfMemoryError(RuntimeError):
    pass


class _FakeKernel:
    def __init__(self, failure):
        self.failure = failure
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, kwargs.copy()))
            error = self.failure(kwargs)
            if isinstance(error, Exception):
                raise error
            if error:
                raise RuntimeError(error)

        return launch


def _direct_case(kind, kernel, allocated, cleanup_before_retry_allocate=None):
    def allocate(shape, *, device, dtype):
        if len(allocated) == 1 and cleanup_before_retry_allocate is not None:
            cleanup_before_retry_allocate.append(allocated[0]() is None)
        output = _FakeOutput(shape)
        allocated.append(weakref.ref(output))
        return output

    names = {
        "_RESOURCE_SAFE_LAUNCH_CACHE",
        "_RESOURCE_SAFE_LAUNCH_CACHE_MAX",
        "_RESOURCE_SAFE_LAUNCH_CACHE_FIFO",
        "_shared_memory_resource_limit",
        "_capture_launch_failure",
        "_rebuild_launch_error",
        "_resource_safe_cache_key",
        "_resource_safe_launch_params",
        "_remember_resource_safe_launch_params",
        "_fused_quant_scaled_mm_direct_call",
        "_scaled_int8_mm_direct_call",
    }
    namespace = _extract(KERNELS / "quanto_int8_inject.py", names)
    namespace.update(
        {
            "torch": SimpleNamespace(empty=allocate),
            "_TRITON_MODULE": SimpleNamespace(
                triton=_FakeTriton(),
                _fused_dynamic_int8_blockscale_gemm_kernel=kernel,
                _scaled_int8_gemm_kernel=kernel,
            ),
            "_TRITON_DIRECT_FUSED_READY": kind == "fused",
            "_TRITON_DIRECT_SCALED_READY": kind == "scaled",
            "_fused_launch_params": lambda *args: (64, 256, 64, 8, 4, 1350, 48),
            "_scaled_launch_params": lambda *args: (64, 256, 64, 8, 4, 1350, 48),
        }
    )
    if kind == "fused":
        call = namespace["_fused_quant_scaled_mm_direct_call"]
        args = (
            _FakeTensor((86400, 4096)),
            _FakeTensor((12288, 4096)),
            object(),
            "bf16",
        )
    else:
        call = namespace["_scaled_int8_mm_direct_call"]
        args = (
            _FakeTensor((86400, 4096)),
            _FakeTensor((12288, 4096)),
            object(),
            object(),
            "bf16",
        )
    return namespace, call, args


class TestLargeMKernelCompatibility(unittest.TestCase):
    def test_shared_memory_classifier_excludes_other_resource_failures(self):
        classify = _extract(
            KERNELS / "quanto_int8_inject.py",
            {"_shared_memory_resource_limit"},
        )["_shared_memory_resource_limit"]
        self.assertTrue(
            classify(
                RuntimeError(
                    "out of resource: shared memory, Required: 102400, "
                    "Hardware limit: 101376"
                )
            )
        )
        for message in (
            "out of resource: registers, Required: 65536, Hardware limit: 65536",
            "CUDA out of memory. Tried to allocate 675 MiB",
            "shared memory compilation error without a resource limit",
        ):
            with self.subTest(message=message):
                self.assertFalse(classify(RuntimeError(message)))

    def test_shared_memory_retry_and_cache_cover_fused_and_scaled_wrappers(self):
        for kind in ("fused", "scaled"):
            with self.subTest(kind=kind):
                allocated = []
                cleanup_observed = []
                cleanup_before_retry_allocate = []

                def fail_wide_tile(kwargs):
                    if kwargs["block_n"] == 128:
                        cleanup_observed.append(allocated[0]() is None)
                    if kwargs["block_n"] == 256:
                        return (
                            "out of resource: shared memory, Required: 102400, "
                            "Hardware limit: 101376"
                        )
                    return None

                kernel = _FakeKernel(
                    fail_wide_tile
                )
                namespace, call, args = _direct_case(
                    kind,
                    kernel,
                    allocated,
                    cleanup_before_retry_allocate,
                )

                first_result = call(*args)
                self.assertEqual(
                    [
                        (grid, kwargs["block_n"], kwargs["block_k"])
                        for grid, kwargs in kernel.calls
                    ],
                    [((1350, 48), 256, 64), ((1350, 96), 128, 64)],
                )
                self.assertEqual(first_result.shape, (86400, 12288))
                self.assertEqual(len(allocated), 2)
                self.assertIsNone(allocated[0]())
                self.assertIs(allocated[1](), first_result)
                self.assertEqual(cleanup_before_retry_allocate, [True])
                self.assertEqual(cleanup_observed, [True])

                second_result = call(*args)
                self.assertEqual(len(kernel.calls), 3)
                self.assertEqual(kernel.calls[-1][0], (1350, 96))
                self.assertEqual(kernel.calls[-1][1]["block_n"], 128)
                self.assertEqual(second_result.shape, (86400, 12288))
                self.assertIs(allocated[-1](), second_result)
                self.assertTrue(namespace["_RESOURCE_SAFE_LAUNCH_CACHE"])

    def test_failed_shared_memory_retry_releases_outputs_without_caching(self):
        failure = (
            "out of resource: shared memory, Required: 102400, "
            "Hardware limit: 101376"
        )
        for kind in ("fused", "scaled"):
            with self.subTest(kind=kind):
                allocated = []
                cleanup_before_retry_allocate = []
                kernel = _FakeKernel(lambda kwargs: failure)
                namespace, call, args = _direct_case(
                    kind,
                    kernel,
                    allocated,
                    cleanup_before_retry_allocate,
                )

                with self.assertRaisesRegex(
                    RuntimeError, "failed after shared-memory retry"
                ) as raised:
                    call(*args)

                self.assertEqual(len(kernel.calls), 2)
                self.assertEqual(len(allocated), 2)
                self.assertEqual(cleanup_before_retry_allocate, [True])
                self.assertIsNone(allocated[0]())
                self.assertIsNone(allocated[1]())
                self.assertEqual(namespace["_RESOURCE_SAFE_LAUNCH_CACHE"], {})
                self.assertIsInstance(raised.exception.__cause__, RuntimeError)
                self.assertIn(failure, str(raised.exception.__cause__))

    def test_cuda_oom_does_not_trigger_smaller_tile_retry(self):
        for kind in ("fused", "scaled"):
            with self.subTest(kind=kind):
                allocated = []
                kernel = _FakeKernel(
                    lambda kwargs: _FakeOutOfMemoryError(
                        "CUDA out of memory. Tried to allocate 675 MiB"
                    )
                )
                namespace, call, args = _direct_case(kind, kernel, allocated)
                with self.assertRaisesRegex(
                    RuntimeError, "CUDA out of memory"
                ) as raised:
                    call(*args)
                self.assertEqual(len(kernel.calls), 1)
                self.assertEqual(len(allocated), 1)
                self.assertIsNone(allocated[0]())
                self.assertEqual(namespace["_RESOURCE_SAFE_LAUNCH_CACHE"], {})
                self.assertIsInstance(
                    raised.exception.__cause__, _FakeOutOfMemoryError
                )


if __name__ == "__main__":
    unittest.main()
