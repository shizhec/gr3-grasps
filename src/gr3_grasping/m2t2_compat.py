"""Narrow, recorded compatibility fixes for the pinned upstream M2T2 code."""
import hashlib
import inspect
import textwrap
from types import MethodType


def correct_object_mask_filter(model):
    """Bind upstream infer with one missing pair of parentheses restored.

    Upstream evaluates ``boolean & integer_count > 0``. This keeps only odd
    mask sizes, even when objectness is high. Preserve the rest of the pinned
    implementation and weights; never edit the downloaded source checkout.
    Fail if upstream changes, instead of silently applying an unverified fix.
    """
    original = type(model).infer
    source = textwrap.dedent(inspect.getsource(original))
    before = "(score > cfg.object_thresh) & mask.sum(dim=1) > 0"
    after = "(score > cfg.object_thresh) & (mask.sum(dim=1) > 0)"
    if source.count(before) != 1:
        raise RuntimeError("Pinned M2T2 object-mask filter changed; review compatibility fix")
    corrected = source.replace(before, after)
    namespace = {}
    exec(compile(corrected, "<gr3_m2t2_object_mask_filter_fix>", "exec"), original.__globals__, namespace)
    model.infer = MethodType(namespace[original.__name__], model)
    return {"name": "object_mask_filter_parentheses", "before": before, "after": after,
            "original_infer_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "corrected_infer_sha256": hashlib.sha256(corrected.encode()).hexdigest()}
