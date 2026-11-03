"""Offline stage comparisons with logical cache mapping and bounded evidence."""
import math

from vllm_bench.mtp_debug.attention_layout import bytes_difference, index_differences, split_packed
from vllm_bench.mtp_debug.tensor_io import WIDTH, compare_bytes, numbers, read_tensor


class Reader:
    def __init__(self, root):
        self.root = root
        self.cache = {}

    def read(self, descriptor):
        if not descriptor:
            raise ValueError("payload absent")
        key = (descriptor.get("path"), descriptor.get("sha256"))
        if key not in self.cache:
            self.cache[key] = read_tensor(self.root, descriptor)
        return self.cache[key]

    def row(self, descriptor, index):
        raw = self.read(descriptor)
        shape = descriptor["shape"]
        if not shape or not 0 <= index < shape[0]:
            raise ValueError("payload row out of range")
        width = math.prod(shape[1:]) * WIDTH[descriptor["dtype"]]
        return raw[index * width:(index + 1) * width], shape[1:]

    def numeric(self, descriptor):
        return numbers(self.read(descriptor), descriptor["dtype"])


def compare_data(a, b, dtype, shape, opaque=False):
    return {"status": "compared", **(bytes_difference(a, b) if opaque else compare_bytes(a, b, dtype, shape))}


def compare_payload(reader, a, b, row_a=None, row_b=None):
    try:
        if row_a is None:
            raw_a, raw_b = reader.read(a), reader.read(b)
            shape_a, shape_b = a["shape"], b["shape"]
        else:
            raw_a, shape_a = reader.row(a, row_a)
            raw_b, shape_b = reader.row(b, row_b)
        if a["dtype"] != b["dtype"] or shape_a != shape_b:
            return {"status": "incompatible_shape_or_dtype", "left_shape": shape_a, "right_shape": shape_b}
        opaque = a.get("comparison_mode") == "bytes_only" or b.get("comparison_mode") == "bytes_only"
        return compare_data(raw_a, raw_b, a["dtype"], shape_a, opaque)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {"status": "unavailable", "reason": str(error)}


def compare_paged(reader, a, b, position, selected_a=None, selected_b=None, packed=False):
    try:
        if not a or not b:
            raise ValueError("paged payload absent")
        pos_a, pos_b = a["logical_positions"], b["logical_positions"]
        ia, ib = {p: i for i, p in enumerate(pos_a)}, {p: i for i, p in enumerate(pos_b)}
        if len(ia) != len(pos_a) or len(ib) != len(pos_b):
            raise ValueError("ambiguous duplicate logical cache positions")
        wanted_a = {p for p in (selected_a if selected_a is not None else pos_a) if 0 <= p <= position}
        wanted_b = {p for p in (selected_b if selected_b is not None else pos_b) if 0 <= p <= position}
        common = sorted(wanted_a & wanted_b & ia.keys() & ib.keys())
        full = wanted_a <= ia.keys() and wanted_b <= ib.keys()
        raw_a, raw_b = [], []
        for p in common:
            x, shape_a = reader.row(a, ia[p])
            y, shape_b = reader.row(b, ib[p])
            if shape_a != shape_b or a["dtype"] != b["dtype"]:
                raise ValueError("paged row layouts differ")
            raw_a.append(x)
            raw_b.append(y)
        if not common:
            # Disjoint selections are an observed index difference, not missing
            # capture. Still validate both payloads before claiming coverage.
            reader.read(a)
            reader.read(b)
            return {"status": "compared", "reason": "no common causal logical positions", "coverage_complete": full,
                    "compared_positions": [], "same_selected_positions": wanted_a == wanted_b,
                    "exact_equal": wanted_a == wanted_b, "left_only_positions": sorted(wanted_a - wanted_b),
                    "right_only_positions": sorted(wanted_b - wanted_a)}
        x, y = b"".join(raw_a), b"".join(raw_b)
        result = {"status": "compared", "coverage_complete": full, "compared_positions": common,
                  "same_selected_positions": wanted_a == wanted_b,
                  "left_only_positions": sorted(wanted_a - wanted_b), "right_only_positions": sorted(wanted_b - wanted_a)}
        if packed:
            parts_a, parts_b = [split_packed(row) for row in raw_a], [split_packed(row) for row in raw_b]
            result["components"] = {
                "latent": compare_data(b"".join(p["latent"] for p in parts_a), b"".join(p["latent"] for p in parts_b), "uint8", [len(common), 512], True),
                "rotary": compare_data(b"".join(p["rotary"] for p in parts_a), b"".join(p["rotary"] for p in parts_b), "bfloat16", [len(common), 64]),
                "scales": compare_data(b"".join(p["scales"] for p in parts_a), b"".join(p["scales"] for p in parts_b), "float32", [len(common), 4])}
            result["exact_equal"] = all(part["exact_equal"] for part in result["components"].values())
        else:
            opaque = a.get("comparison_mode") == "bytes_only" or b.get("comparison_mode") == "bytes_only"
            result.update(compare_data(x, y, a["dtype"], [len(common), *shape_a], opaque))
        result["exact_equal"] &= wanted_a == wanted_b
        return result
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {"status": "unavailable", "reason": str(error), "coverage_complete": False}


def stage_map(layer):
    result = {}
    for stage in layer.get("stages", []):
        result.setdefault(stage["name"], []).append(stage)
    return {key: value[0] for key, value in result.items() if len(value) == 1}


def tensor_row(stage, position):
    rows = stage.get("row_positions", [])
    if rows.count(position) != 1:
        raise ValueError("query position absent or ambiguous")
    return rows.index(position)


def compare_layer(reader, a, b, position):
    """One query position, same rank and same committed prefix."""
    records = []
    stages_a, stages_b = stage_map(a), stage_map(b)

    def add(boundary, field, result):
        records.append({"boundary": boundary, "field": field, **result})

    def control(kind, field):
        ca = [call for call in a.get("mome_controls", []) if call["prefix"].endswith(kind)]
        cb = [call for call in b.get("mome_controls", []) if call["prefix"].endswith(kind)]
        if len(ca) != 1 or len(cb) != 1:
            return {"status": "unavailable", "reason": "MoME control absent/ambiguous"}
        x, y = ca[0].get("tensor_capture", {}), cb[0].get("tensor_capture", {})
        try:
            ra, rb = tensor_row(x, position), tensor_row(y, position)
            return compare_payload(reader, x.get(field), y.get(field), ra, rb)
        except ValueError as error:
            return {"status": "unavailable", "reason": str(error)}

    def control_parameters(kind):
        calls = [[c for c in layer.get("mome_controls", []) if c["prefix"].endswith(kind)] for layer in (a, b)]
        if any(len(c) != 1 for c in calls):
            return {"status": "unavailable", "reason": "MoME control absent/ambiguous"}
        operands = []
        for (call,) in calls:
            parameters = call.get("tensor_capture", {}).get("parameters")
            if not parameters or not parameters.get("weight"):
                return {"status": "unavailable", "reason": "MoME weights/parameters absent"}
            metadata = {k: v for k, v in call.get("metadata", {}).items() if k in
                        ("num_reqs", "num_actual_tokens", "num_decodes", "num_prefills", "max_query_len",
                         "B_size", "seq_lens", "query_start_loc", "num_accepted_tokens", "num_computed_tokens")}
            operands.append({"parameters": parameters, "metadata": metadata})
        return {"status": "compared", "exact_equal": operands[0] == operands[1], "left": operands[0], "right": operands[1]}

    for kind, prefix in (("qa_conv", "q_mome"), ("compresskv_conv", "kv_mome")):
        add(prefix + "_parameters", "parameters", control_parameters(kind))
        add(prefix + "_input", "x_before", control(kind, "x_before"))
        add(prefix + "_output", "y_after", control(kind, "y_after"))
    for stage_name in ("indexer_kernel", "attention_kernel", "projection"):
        x, y = stages_a.get(stage_name), stages_b.get(stage_name)
        if x is None or y is None:
            if stage_name == "indexer_kernel" and a["config"].get("skip_topk") and b["config"].get("skip_topk"):
                continue
            add(stage_name, "stage", {"status": "unavailable", "reason": "operator not observed/ambiguous"})
            continue
        if x.get("status") != "complete" or y.get("status") != "complete":
            add(stage_name, "stage", {"status": "unavailable", "reason": "incomplete operator capture"})
            continue
        try:
            ra, rb = tensor_row(x, position), tensor_row(y, position)
        except ValueError as error:
            add(stage_name, "alignment", {"status": "unavailable", "reason": str(error)})
            continue
        tx, ty = x["tensors"], y["tensors"]
        required = ({"query", "sparse_indices", "key_rows", "value", "actual_seq_lengths_query", "actual_seq_lengths_kv", "output"}
                    if stage_name == "attention_kernel" else {"query", "weights", "key_rows", "key_scales", "indices"}
                    if stage_name == "indexer_kernel" else {"input", "output"})
        if stage_name == "attention_kernel" and (a["config"].get("sink_count", 0) or b["config"].get("sink_count", 0)):
            required |= {"key_sink", "value_sink"}
        for field in sorted(required - (tx.keys() & ty.keys())):
            add(stage_name + "_input", field, {"status": "unavailable", "reason": "required kernel operand missing"})
        parameters_a = {"operator": x.get("operator"), "scalars": x.get("scalars"), "weight": x.get("weight")}
        parameters_b = {"operator": y.get("operator"), "scalars": y.get("scalars"), "weight": y.get("weight")}
        add(stage_name + "_parameters", "parameters", {"status": "compared", "exact_equal": parameters_a == parameters_b,
                                                      "left": parameters_a, "right": parameters_b})
        selection_a = selection_b = None
        if stage_name == "attention_kernel":
            try:
                ba, _ = reader.row(tx["sparse_indices"], ra)
                bb, _ = reader.row(ty["sparse_indices"], rb)
                selection_a, selection_b = numbers(ba, tx["sparse_indices"]["dtype"]), numbers(bb, ty["sparse_indices"]["dtype"])
                indices = index_differences(selection_a, selection_b)
                indices.update(status="compared", left_future_positions=sorted({p for p in selection_a if p > position}),
                               right_future_positions=sorted({p for p in selection_b if p > position}))
                add("attention_indices", "sparse_indices", indices)
            except (KeyError, ValueError, OSError) as error:
                add("attention_indices", "sparse_indices", {"status": "unavailable", "reason": str(error)})
        fields = (tx.keys() | ty.keys()) - {"block_table", "sparse_indices"}
        inputs = sorted(fields - {"output", "indices"})
        for field in [*inputs, *sorted(fields & {"output", "indices"})]:
            if field == "key_rows":
                result = compare_paged(reader, tx.get(field), ty.get(field), position, selection_a, selection_b, stage_name == "attention_kernel")
            elif field == "key_scales":
                result = compare_paged(reader, tx.get(field), ty.get(field), position)
            elif field in ("query", "weights", "query_dequant_scale", "query_scale", "output", "indices", "input"):
                result = compare_payload(reader, tx.get(field), ty.get(field), ra, rb)
                if field == "indices" and result["status"] == "compared":
                    ba, _ = reader.row(tx[field], ra)
                    bb, _ = reader.row(ty[field], rb)
                    result.update(index_differences(numbers(ba, tx[field]["dtype"]), numbers(bb, ty[field]["dtype"])))
            else:
                result = compare_payload(reader, tx.get(field), ty.get(field))
            add(stage_name + ("_output" if field in ("output", "indices") else "_input"), field, result)
    add("o_parameters", "parameters", control_parameters("o_conv"))
    add("o_input_after_tp", "x_before", control("o_conv", "x_before"))
    add("o_output", "y_after", control("o_conv", "y_after"))
    return records
