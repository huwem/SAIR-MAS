"""Shared, thread-safe SentenceTransformer loader for TLSG components.

SentenceTransformer initialization is not thread-safe (transformers /
accelerate meta-device init patches are process-global): concurrent lazy
loads from sample worker threads leave modules on the meta device and fail
with "Cannot copy out of meta tensor".  All TLSG components therefore load
through this serialized, process-wide cache — which also means SRAC and
SemanticPolarity share a single model instance.

Environment overrides (same semantics as topology semantic/cosine paths):
    ST_MODEL_PATH    — override the model name/path
    ST_CACHE_FOLDER  — passed as cache_folder
    ST_LOCAL_ONLY    — "1"/"true"/"yes" → local_files_only=True
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model_cache: dict[tuple, object] = {}


def load_sentence_transformer(model_name: str, device: str = "cpu"):
    """Load (or fetch from cache) a SentenceTransformer, serialized process-wide."""
    model_path = os.environ.get("ST_MODEL_PATH", model_name)
    cache_folder = os.environ.get("ST_CACHE_FOLDER", None)
    local_only = os.environ.get("ST_LOCAL_ONLY", "").lower() in ("1", "true", "yes")
    key = (model_path, cache_folder, local_only, device)

    with _model_lock:
        if key not in _model_cache:
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            from sentence_transformers import SentenceTransformer

            logger.debug(
                "Loading SentenceTransformer %s (device=%s, local_only=%s)",
                model_path, device, local_only,
            )
            _model_cache[key] = SentenceTransformer(
                model_path,
                device=device,
                cache_folder=cache_folder,
                local_files_only=local_only,
            )
        return _model_cache[key]
