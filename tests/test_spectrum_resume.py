from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import hofstadter
from hofstadter import SpectrumTask
from model import load_model


ROOT = Path(__file__).resolve().parents[1]


class SpectrumResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(ROOT / "example_square.toml")

    @staticmethod
    def _task(mesh_at_q1: int, resume_from: Path | None = None) -> SpectrumTask:
        return SpectrumTask(
            flux_min=0.0,
            flux_max=0.5,
            q_max=3,
            k_mesh=(1, 1),
            k_mesh_q1=(mesh_at_q1, mesh_at_q1),
            energy_window=(-5.0, 5.0),
            filling_window=(0.0, 1.0),
            dpi=30,
            gap_threshold=0.02,
            resume_from=resume_from,
        )

    def _run(
        self,
        root: Path,
        task: SpectrumTask,
        counter: list[int] | None = None,
    ) -> Path:
        original = hofstadter.magnetic_energies

        def counted(*args: object, **kwargs: object) -> np.ndarray:
            points = np.asarray(args[3])
            if counter is not None:
                counter.append(1 if points.ndim == 1 else points.shape[0])
            return original(*args, **kwargs)

        with (
            mock.patch.object(hofstadter, "_DATA_DIR", root / "data"),
            mock.patch.object(hofstadter, "_FIGURE_DIR", root / "figure"),
            mock.patch.object(hofstadter, "magnetic_energies", side_effect=counted),
        ):
            return hofstadter.run_spectrum(self.model, task)[0]

    def assert_archives_equal(self, first: Path, second: Path) -> None:
        with np.load(first, allow_pickle=False) as left, np.load(
            second, allow_pickle=False
        ) as right:
            self.assertEqual(left.files, right.files)
            for key in left.files:
                left_value = left[key]
                right_value = right[key]
                if left_value.dtype.kind in "fc":
                    np.testing.assert_allclose(
                        left_value,
                        right_value,
                        rtol=0.0,
                        atol=1.0e-12,
                        err_msg=key,
                    )
                else:
                    np.testing.assert_array_equal(
                        left_value,
                        right_value,
                        err_msg=key,
                    )

    def test_incremental_matches_fresh_and_reuses_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_path = self._run(root / "old", self._task(2))
            fresh_path = self._run(root / "fresh", self._task(4))

            incremental_calls: list[int] = []
            incremental_task = self._task(4, resume_from=old_path)
            incremental_path = self._run(
                root / "incremental",
                incremental_task,
                incremental_calls,
            )
            self.assertEqual(sum(incremental_calls), 18)
            self.assert_archives_equal(fresh_path, incremental_path)

            completed_calls: list[int] = []
            completed_path = self._run(
                root / "incremental",
                incremental_task,
                completed_calls,
            )
            self.assertEqual(sum(completed_calls), 0)
            self.assert_archives_equal(fresh_path, completed_path)

    def test_legacy_resume_is_validated_and_matches_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_path = self._run(root / "old", self._task(2))
            fresh_path = self._run(root / "fresh", self._task(4))
            legacy_path = root / "legacy.npz"
            with np.load(old_path, allow_pickle=False) as source:
                legacy_arrays = {
                    key: source[key]
                    for key in source.files
                    if key not in {"spectrum_data_version", "model_fingerprint"}
                }
            np.savez_compressed(legacy_path, **legacy_arrays)

            calls: list[int] = []
            resumed_path = self._run(
                root / "legacy-resume",
                self._task(4, resume_from=legacy_path),
                calls,
            )
            # Four zero-field momenta plus one magnetic momentum validate the
            # legacy model; the remaining 18 calls are the truly missing mesh.
            self.assertEqual(sum(calls), 23)
            self.assert_archives_equal(fresh_path, resumed_path)

    def test_fingerprint_mismatch_is_rejected_before_diagonalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = self._run(root / "source", self._task(2))
            corrupt_path = root / "wrong-model.npz"
            with np.load(source_path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            arrays["model_fingerprint"] = "0" * 64
            np.savez_compressed(corrupt_path, **arrays)

            calls: list[int] = []
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self._run(
                    root / "rejected",
                    self._task(4, resume_from=corrupt_path),
                    calls,
                )
            self.assertEqual(sum(calls), 0)

    def test_parser_resolves_resume_path_and_rejects_negative_threshold(self) -> None:
        source = (ROOT / "example_square.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.toml"
            valid.write_text(
                source
                + '\ngap_threshold = 0.025\nresume_from = "seed.npz"\n',
                encoding="utf-8",
            )
            task = hofstadter.load_hofstadter_task(valid)
            self.assertIsInstance(task, SpectrumTask)
            self.assertEqual(task.gap_threshold, 0.025)
            self.assertEqual(task.resume_from, (root / "seed.npz").resolve())

            invalid = root / "invalid.toml"
            invalid.write_text(
                source + "\ngap_threshold = -0.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                hofstadter.load_hofstadter_task(invalid)

    def test_omitted_gap_threshold_preserves_previous_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = replace(self._task(1), gap_threshold=None)
            data_path = self._run(Path(temporary), task)
            with np.load(data_path, allow_pickle=False) as data:
                self.assertEqual(float(data["wannier_gap_threshold"]), 0.01)


if __name__ == "__main__":
    unittest.main()
