"""QThread workers that call into the RVC backend without blocking the GUI."""

import importlib
import traceback

from PyQt6.QtCore import QThread, pyqtSignal


def _backend():
    """Lazy-import the heavy backend module (infer-web.py) once."""
    return importlib.import_module("infer-web")


class BackendWorker(QThread):
    """Generic worker: runs *func* with *args* / *kwargs* in a background thread.

    If *is_generator* is ``True`` the function is iterated and each yielded
    value is emitted via :pyqtSignal:`log`.  Otherwise the return value is
    emitted once via :pyqtSignal:`result`.
    """

    log = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, args=(), kwargs=None, is_generator=False, parent=None):
        super().__init__(parent)
        self._func = func
        self._args = args
        self._kwargs = kwargs or {}
        self._is_generator = is_generator

    def run(self):
        try:
            ret = self._func(*self._args, **self._kwargs)
            if self._is_generator:
                last = None
                for msg in ret:
                    last = msg
                    self.log.emit(str(msg))
                self.result.emit(last)
            else:
                self.result.emit(ret)
        except Exception:
            self.error.emit(traceback.format_exc())


class ModelLoadWorker(QThread):
    """Load a voice model via the backend ``get_vc`` function."""

    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_name, protect0=0.33, protect1=0.33, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._protect0 = protect0
        self._protect1 = protect1

    def run(self):
        try:
            backend = _backend()
            res = backend.get_vc(self._model_name, self._protect0, self._protect1)
            self.result.emit(res)
        except Exception:
            self.error.emit(traceback.format_exc())


class InferenceWorker(QThread):
    """Run single-file voice conversion."""

    result = pyqtSignal(str, object)  # info_text, (sr, audio_np) | (None, None)
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._p = params

    def run(self):
        try:
            backend = _backend()
            info, audio = backend.vc_single(
                self._p["sid"],
                self._p["input_audio_path0"],
                self._p["input_audio_path1"],
                self._p["f0_up_key"],
                self._p["f0_file"],
                self._p["f0_method"],
                self._p["file_index"],
                self._p["file_index2"],
                self._p["index_rate"],
                self._p["filter_radius"],
                self._p["resample_sr"],
                self._p["rms_mix_rate"],
                self._p["protect"],
                self._p["crepe_hop_length"],
            )
            self.result.emit(info, audio)
        except Exception:
            self.error.emit(traceback.format_exc())
