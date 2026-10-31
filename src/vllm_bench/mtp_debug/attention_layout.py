"""Pure logical-page mapping and the verified 656-byte HiF8 cache layout."""
from itertools import islice


PACKED_WIDTH = 656


def flatten(value):
    if isinstance(value, (tuple, list)):
        return [x for item in value for x in flatten(item)]
    return [value]


def selected_positions(indices, length):
    raw = flatten(indices)
    valid = sorted({int(i) for i in raw if 0 <= i < length})
    invalid = sorted({int(i) for i in raw if i != -1 and not 0 <= i < length})
    return valid, {"padding_count": raw.count(-1), "invalid_indices": invalid}


def page_coordinates(positions, table, block_size, page_count):
    if len(table) != 1 or block_size < 1:
        raise ValueError("Only single-request paged layout supported")
    pages, slots = [], []
    for position in positions:
        logical_page, offset = divmod(position, block_size)
        if position < 0 or logical_page >= len(table[0]):
            raise ValueError("Logical KV position outside block table")
        physical = table[0][logical_page]
        if not 0 <= physical < page_count:
            raise ValueError("Physical KV page outside cache")
        pages.append(physical)
        slots.append(offset)
    return pages, slots


def split_packed(row):
    if len(row) != PACKED_WIDTH:
        raise ValueError("Expected 512 HiF8 + 128 BF16 rope bytes + 16 FP32 scale bytes")
    return {"latent": row[:512], "rotary": row[512:640], "scales": row[640:656]}


def index_differences(left, right):
    a, b = flatten(left), flatten(right)
    sa, sb = set(a) - {-1}, set(b) - {-1}
    return {"exact_equal": a == b, "same_valid_set": sa == sb,
            "only_order_or_padding_changed": a != b and sa == sb,
            "left_only_positions": sorted(sa - sb), "right_only_positions": sorted(sb - sa),
            "left_duplicates": len([i for i in a if i != -1]) - len(sa),
            "right_duplicates": len([i for i in b if i != -1]) - len(sb),
            "left_padding": a.count(-1), "right_padding": b.count(-1)}


def bytes_difference(left, right):
    if left == right:
        return {"exact_equal": True, "different_bytes": 0, "left_nbytes": len(left), "right_nbytes": len(right), "first_different_offsets": []}
    return {"exact_equal": left == right, "different_bytes": sum(x != y for x, y in zip(left, right)) + abs(len(left) - len(right)),
            "left_nbytes": len(left), "right_nbytes": len(right),
            "first_different_offsets": list(islice((i for i, (x, y) in enumerate(zip(left, right)) if x != y), 16))}
