"""Exact tensor payloads and numerical comparisons; offline code is stdlib-only."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


WIDTH = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8,
         "uint8": 1, "int8": 1, "int32": 4, "int64": 8}
CODES = {"float16": "e", "float32": "f", "float64": "d", "uint8": "B", "int8": "b", "int32": "i", "int64": "q"}


def numbers(raw, dtype):
    if dtype not in WIDTH or len(raw) % WIDTH[dtype]:
        raise ValueError(f"Unsupported dtype or invalid byte length: {dtype}")
    if dtype == "bfloat16":
        return [struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                for (bits,) in struct.iter_unpack("<H", raw)]
    return [value for (value,) in struct.iter_unpack("<" + CODES[dtype], raw)]


def json_number(value):
    return value if math.isfinite(value) else str(value)


def json_safe(value):
    if isinstance(value, float):
        return json_number(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def tensor_bytes(tensor):
    """Materialize immutable bytes before the wrapped function can mutate x/cache."""
    import torch
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in WIDTH:
        raise ValueError(f"Unsupported tensor dtype: {dtype}")
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    if sys.byteorder != "little":
        width = WIDTH[dtype]
        raw = b"".join(raw[i:i + width][::-1] for i in range(0, len(raw), width))
    return raw


def tensor_identity(tensor):
    raw = tensor_bytes(tensor)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(raw).hexdigest()}


class TensorWriter:
    def __init__(self, root, pid, rank=0, max_tensor_bytes=16 * 1024**2, budget_bytes=512 * 1024**2):
        self.root = Path(root)
        self.pid, self.rank = pid, rank
        self.max_tensor_bytes, self.budget_bytes = max_tensor_bytes, budget_bytes
        self.written = 0
        self.stopped_reason = None
        self.write_status()

    def write_status(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"tensor-status-rank-{self.rank}-pid-{self.pid}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"format": "pangu-mome-tensor-status-v1",
            "status": "stopped" if self.stopped_reason else "recording",
            "reason": self.stopped_reason, "bytes_written": self.written,
            "max_tensor_bytes": self.max_tensor_bytes, "budget_bytes": self.budget_bytes}) + "\n")
        temporary.replace(path)

    def can_save(self, size):
        if self.stopped_reason:
            return False
        if size > self.max_tensor_bytes or self.written + size > self.budget_bytes:
            self.stopped_reason = "tensor_size_limit" if size > self.max_tensor_bytes else "run_byte_budget"
            self.write_status()
            return False
        return True

    def save(self, tensor, forward_index, call_index, name, row_summaries=True):
        if tensor is None:
            return {"status": "not_recorded", "reason": "no_tensor_value"}
        size = tensor.numel() * tensor.element_size()
        if not self.can_save(size):
            return {"status": "not_recorded", "reason": self.stopped_reason,
                    "requested_bytes": size}
        path = self.root / "tensors" / f"rank-{self.rank}-pid-{self.pid}" / f"forward-{forward_index:06d}" / f"call-{call_index:03d}" / f"{name}.bin"
        temporary = path.with_suffix(".tmp")
        try:
            raw = tensor_bytes(tensor)
            dtype = str(tensor.dtype).removeprefix("torch.")
            shape = list(tensor.shape)
            if dtype not in WIDTH or len(raw) != size or math.prod(shape) * WIDTH[dtype] != size:
                raise ValueError("Tensor shape/dtype/byte size mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Output directories are new; refuse to overwrite an earlier call's payload.
            if path.exists():
                raise FileExistsError(path)
            with temporary.open("xb") as stream:
                stream.write(raw)
            temporary.replace(path)
            self.written += size
            rows = []
            if row_summaries and shape and shape[0]:
                row_bytes = size // shape[0]
                for index in range(shape[0]):
                    content = raw[index * row_bytes:(index + 1) * row_bytes]
                    rows.append({"row": index, "sha256": hashlib.sha256(content).hexdigest(),
                                 "sample": [json_number(x) for x in numbers(content[:16 * WIDTH[dtype]], dtype)]})
            result = {"status": "complete", "path": path.relative_to(self.root).as_posix(),
                      "shape": shape, "dtype": dtype, "byteorder": "little",
                      "nbytes": size, "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows,
                      "row_summaries": row_summaries,
                      "original_strides": list(tensor.stride()) if hasattr(tensor, "stride") else None}
            self.write_status()
            return result
        except Exception:
            self.stopped_reason = "capture_error"
            temporary.unlink(missing_ok=True)
            try:
                self.write_status()
            except OSError:
                pass
            raise


def read_tensor(root, descriptor):
    if not descriptor or descriptor.get("status") != "complete":
        raise ValueError("Tensor was not recorded completely")
    root = Path(root).resolve()
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Tensor path escapes capture directory")
    dtype, shape = descriptor["dtype"], descriptor["shape"]
    if dtype not in WIDTH or descriptor.get("byteorder") != "little":
        raise ValueError("Unsupported dtype or byte order")
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid tensor shape")
    size = math.prod(shape) * WIDTH[dtype]
    if size != descriptor["nbytes"] or path.stat().st_size != size:
        raise ValueError("Tensor file size does not match descriptor")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Tensor SHA256 mismatch")
    return raw


def coordinates(index, shape):
    result = []
    for extent in reversed(shape):
        result.append(index % extent)
        index //= extent
    return list(reversed(result))


def compare_bytes(left, right, dtype, shape):
    """Exact bit differences plus finite-only numerical metrics; no NaN equality claims."""
    width = WIDTH[dtype]
    if len(left) != len(right) or len(left) != math.prod(shape) * width:
        raise ValueError("Cannot compare mismatched shapes/byte lengths")
    a, b = numbers(left, dtype), numbers(right, dtype)
    count = 0
    max_abs = 0.0
    max_index = None
    first = []
    nonfinite = {"left": 0, "right": 0}
    sum_delta2 = sum_a2 = sum_b2 = 0.
    max_ulp = None
    for index, (x, y) in enumerate(zip(a, b)):
        nonfinite["left"] += not math.isfinite(x)
        nonfinite["right"] += not math.isfinite(y)
        unequal = left[index * width:(index + 1) * width] != right[index * width:(index + 1) * width]
        if unequal:
            count += 1
            if len(first) < 16:
                first.append({"index": coordinates(index, shape), "left": json_number(x), "right": json_number(y)})
        if math.isfinite(x) and math.isfinite(y):
            delta = abs(x - y)
            try:
                sum_delta2 += delta * delta
                sum_a2 += x * x
                sum_b2 += y * y
            except OverflowError:
                sum_delta2 = float("inf")
            if dtype in ("bfloat16", "float16", "float32", "float64"):
                sign = 1 << (width * 8 - 1)
                mask = (1 << (width * 8)) - 1
                def ordered(content):
                    bits = int.from_bytes(content, "little")
                    return (~bits & mask) if bits & sign else bits | sign
                ulp = abs(ordered(left[index * width:(index + 1) * width]) - ordered(right[index * width:(index + 1) * width]))
                max_ulp = max(max_ulp or 0, ulp)
            if delta > max_abs:
                max_abs, max_index = delta, coordinates(index, shape)
    result = {"exact_equal": left == right, "different_elements": count,
            "max_abs_finite": json_number(max_abs), "max_abs_index": max_index,
            "nonfinite_elements": nonfinite, "first_differences": first}
    result.update(relative_l2_finite=json_number(math.sqrt(sum_delta2) / max(math.sqrt(sum_a2), math.sqrt(sum_b2), 1e-12)),
                  max_ulp_finite=max_ulp)
    return result


def compare_tensors(left_root, left, right_root, right, rows=None):
    """Missing/corrupt payloads are unavailable, never treated as equal."""
    try:
        a, b = read_tensor(left_root, left), read_tensor(right_root, right)
        shape_a, shape_b = left["shape"], right["shape"]
        compatible = shape_a == shape_b if rows is None else bool(shape_a and shape_b and shape_a[1:] == shape_b[1:])
        if not compatible or left["dtype"] != right["dtype"]:
            return {"status": "incompatible_shape_or_dtype"}
        shape, dtype = left["shape"], left["dtype"]
        if rows is not None:
            row_a, row_b = rows
            if not shape or not 0 <= row_a < shape_a[0] or not 0 <= row_b < shape_b[0]:
                raise ValueError("Row index outside tensor")
            length = math.prod(shape[1:]) * WIDTH[dtype]
            a, b = a[row_a * length:(row_a + 1) * length], b[row_b * length:(row_b + 1) * length]
            shape = shape[1:]
        return {"status": "compared", **compare_bytes(a, b, dtype, shape)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "unavailable", "reason": str(error)}
