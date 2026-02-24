"""Compatibility shim for radgraph with transformers >= 4.46.

radgraph bundles a vendored allennlp that calls several methods removed
(or moved) in recent transformers releases.  Call
``patch_transformers_for_radgraph()`` once before instantiating F1RadGraph.

Methods restored
----------------
encode_plus
    Removed in transformers 4.46; replaced by __call__.
    Shim: delegate to __call__ with the same arguments.

build_inputs_with_special_tokens
    Defined on PreTrainedTokenizer (slow) but absent on
    PreTrainedTokenizerFast / PreTrainedTokenizerBase in newer releases.
    Shim: prepend cls_token_id and append sep_token_id (BERT convention),
    falling back to returning the bare ids when these special tokens
    are not available.

num_special_tokens_to_add
    Used by allennlp to compute sequence lengths.
    Shim: returns 2 when cls+sep tokens exist, 0 otherwise.
"""

import logging

logger = logging.getLogger(__name__)


def patch_transformers_for_radgraph() -> None:
    """Monkey-patch PreTrainedTokenizerBase to restore allennlp-required methods."""
    try:
        from transformers import PreTrainedTokenizerBase as _PTTB
    except ImportError:
        return  # transformers not installed — nothing to patch

    patched: list[str] = []

    # ── encode_plus ──────────────────────────────────────────────────────────
    if not hasattr(_PTTB, "encode_plus"):
        def _encode_plus(self, text, text_pair=None, **kw):
            return self(text, text_pair=text_pair, **kw)
        _PTTB.encode_plus = _encode_plus
        patched.append("encode_plus")

    # ── build_inputs_with_special_tokens ─────────────────────────────────────
    if not hasattr(_PTTB, "build_inputs_with_special_tokens"):
        def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            cls_id = getattr(self, "cls_token_id", None)
            sep_id = getattr(self, "sep_token_id", None)
            cls = [cls_id] if cls_id is not None else []
            sep = [sep_id] if sep_id is not None else []
            ids_0 = list(token_ids_0)
            if token_ids_1 is None:
                return cls + ids_0 + sep
            return cls + ids_0 + sep + list(token_ids_1) + sep
        _PTTB.build_inputs_with_special_tokens = _build_inputs_with_special_tokens
        patched.append("build_inputs_with_special_tokens")

    # ── num_special_tokens_to_add ─────────────────────────────────────────────
    if not hasattr(_PTTB, "num_special_tokens_to_add"):
        def _num_special_tokens_to_add(self, pair: bool = False) -> int:
            has_cls = getattr(self, "cls_token_id", None) is not None
            has_sep = getattr(self, "sep_token_id", None) is not None
            if has_cls and has_sep:
                return 3 if pair else 2
            return 0
        _PTTB.num_special_tokens_to_add = _num_special_tokens_to_add
        patched.append("num_special_tokens_to_add")

    if patched:
        logger.debug(
            "radgraph_compat: patched PreTrainedTokenizerBase — %s",
            ", ".join(patched),
        )
