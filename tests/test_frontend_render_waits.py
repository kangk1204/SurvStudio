"""Bounded rendering waits when browsers stop producing animation frames."""
from pathlib import Path
import shutil

import pytest

from test_frontend_review import _run_page, example_dataset  # noqa: F401 (pytest fixture)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the front-end tests")


@pytest.mark.parametrize("deliver_frames", [True, False])
def test_render_frame_wait_completes_and_cleans_up_when_frames_pause(tmp_path: Path, deliver_frames: bool) -> None:
    result = _run_page(tmp_path, r"""
      const callbacks = [], cancelled = [];
      page.context.requestAnimationFrame = (callback) => { callbacks.push(callback); return callbacks.length; };
      page.context.cancelAnimationFrame = (id) => { cancelled.push(id); };
      page.context.__resolved = 0;
      page.run("waitForRenderFrames(2).then(() => { __resolved += 1; })");
      if (fixtures.deliverFrames) {
        callbacks[0]();
        callbacks[1]();
      }
      // The harness advances queued timers, so this exercises the fallback without a real-time sleep.
      await page.settle();
      const completed = page.context.__resolved;
      const requested = callbacks.length;
      // A late callback must not schedule more frames or resolve the wait a second time.
      callbacks.forEach((callback) => callback());
      await page.settle();
      return { completed, afterLate: page.context.__resolved, requested,
        requestedAfterLate: callbacks.length, cancelled: cancelled.length };
    """, deliverFrames=deliver_frames)
    assert result["completed"] == result["afterLate"] == 1
    assert result["requested"] == result["requestedAfterLate"] == (2 if deliver_frames else 1)
    assert result["cancelled"] == 1


def test_hidden_plot_stabilization_does_not_wait_for_animation_frames(tmp_path: Path) -> None:
    result = _run_page(tmp_path, r"""
      page.document.hidden = true;
      page.context.requestAnimationFrame = () => { throw new Error('Hidden pages must not wait for a frame'); };
      page.context.__completed = false;
      page.run("stabilizePlotsBeforeBanner([{ closest: () => null }], {}).then(() => { __completed = true; })");
      await page.settle();
      return page.context.__completed;
    """)
    assert result is True


def test_model_banners_distinguish_restored_results_from_run_duration(tmp_path: Path, example_dataset: dict) -> None:
    result = _run_page(tmp_path, r"""
      await loadDataset(page, fixtures.dataset);
      const config = page.run("currentBaseConfig()");
      const ml = { request_config: { ...config, model_type: 'rsf', features: ['age'] },
        analysis: { model_stats: { c_index: 0.61, n_patients: 360, n_features: 1, evaluation_mode: 'holdout' } } };
      const dl = { request_config: { ...config, model_type: 'deepsurv', features: ['age'] },
        analysis: { c_index: 0.62, epochs_trained: 10, evaluation_mode: 'holdout' } };
      page.context.__ml = ml; page.context.__dl = dl;
      await page.run("renderMlModelResult(__ml)");
      await page.run("renderDlModelResult(__dl)");
      const saved = page.run("[refs.mlMetaBanner.textContent, refs.dlMetaBanner.textContent]");
      await page.run("renderMlModelResult(__ml, { elapsedSeconds: '4.2' })");
      await page.run("renderDlModelResult(__dl, { elapsedSeconds: '5.3' })");
      return { saved, fresh: page.run("[refs.mlMetaBanner.textContent, refs.dlMetaBanner.textContent]") };
    """, dataset=example_dataset)
    assert all(text.endswith(", restored saved result") for text in result["saved"])
    assert result["fresh"][0].endswith(", time=4.2s")
    assert result["fresh"][1].endswith(", time=5.3s")
