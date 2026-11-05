"""Weight packing with layer context for MXFP8 scale alignment failures."""

import json


def packing_context(model, scope):
    linears = []
    for name, layer in model.named_modules():
        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        if weight is None or scale is None or getattr(weight, "is_mxfp8_packed", False):
            continue
        if (len(weight.shape) == len(scale.shape) == 2 and 0 not in weight.shape
                and scale.shape[-1] % 2 and scale.shape[-1] * 32 in weight.shape):
            linears.append({"layer": name, "prefix": getattr(layer, "prefix", name),
                            "layer_key": getattr(layer, "layer_name_inside_block", None),
                            "method": type(getattr(layer, "quant_method", None)).__name__,
                            "tp_size": getattr(layer, "tp_size", None),
                            "weight_shape": list(weight.shape), "scale_shape": list(scale.shape)})
    return {"scope": scope, "unaligned_count": len(linears), "unaligned_linears": linears[:4]}


def pack_weights(model, model_config, device, *, scope):
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    try:
        process_weights_after_loading(model, model_config, device)
    except Exception:
        # Diagnostics must not replace the original operator exception.
        try:
            print("[weight-pack] " + json.dumps(packing_context(model, scope)), flush=True)
        except Exception as error:
            print(f"[weight-pack] scope={scope} diagnostic_error={error}", flush=True)
        raise
