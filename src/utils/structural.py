"""Structural similarity scoring for nested JSON outputs.

Top-level entry point used by the pipeline is `structural_scores(j1, j2)`.
The keys returned depend on the inputs' top-level shape:

  - dict vs dict   -> {alignment, type_consistency, content_fidelity}
  - list vs list   -> {matching, alignment, type_consistency, content_fidelity}
  - mixed / other  -> {} (defensive; only triggers if a caller bypasses
                          schema validation)

The four scores form a coarse-to-fine cascade:

  matching          fraction of dicts in two top-level lists that pair
                    above a similarity threshold. Only defined for list
                    inputs - a single dict-vs-dict input has nothing
                    to pair at the top level (the dicts themselves are
                    the comparison).

  alignment         "structure ratio" - of all key-edges in the dict
                    tree, fraction that exist on both sides. Recursive:
                    accumulates over every level of dict-vs-dict descent.

  type_consistency  "type ratio" - at the leaves of recursion (where a
                    leaf is anything that is not a dict-vs-dict pair),
                    fraction with equal Python types.

  content_fidelity  "content ratio" - among leaves with equal types,
                    fraction with equal values (Python `==`).
"""

from typing import Any, Literal

from pydantic import BaseModel

# Sum-of-three-ratios threshold for accepting a dict pair when matching
# elements between two lists. Out of a maximum of 3.0; permissive enough
# that a structurally-and-type-aligned pair clears even when no values agree.
PAIR_THRESHOLD: float = 1.75


# Marker emitted by `compare_dict_structure` for non-dict-vs-non-dict leaves.
# T = matching value, F = same type/different value, X = type mismatch.
LeafMarker = Literal["T", "F", "X"]


class DictRatios(BaseModel):
    """Three normalized ratios for a dict-vs-dict comparison.

    Field names match the keys exposed in `StructuralScores` so the
    pipeline-facing output reuses them without translation.
    """

    alignment: float
    type_consistency: float
    content_fidelity: float

    def total(self) -> float:
        """
        Sum the three ratios.

        Returns
        -------
        float
            `alignment + type_consistency + content_fidelity`. Used by
            `compare_list_structure` against `PAIR_THRESHOLD`.
        """
        return self.alignment + self.type_consistency + self.content_fidelity


class DictCounters(BaseModel):
    """Six raw counters produced by a recursive dict-vs-dict descent.

    `edge_*` apply at every level of the dict tree (counts of
    matched/unmatched keys); `type_*` and `node_*` apply at the leaves
    of recursion only (where a leaf is anything that is not a
    dict-vs-dict pair).
    """

    edge_equal: int
    edge_diff: int
    type_equal: int
    type_diff: int
    node_equal: int
    node_diff: int


class DictCountTree(BaseModel):
    """Raw count tree returned by `compare_dict_structure` for a dict pair.

    `nested` interleaves child trees (further dict pairs) with leaf
    markers ("T"/"F"/"X") in the order keys appear in the left-hand dict.
    """

    matched: int
    unmatched: int
    nested: list["DictCountTree | LeafMarker"]


class ListComparisonResult(BaseModel):
    """Outcome of pairing dict elements between two lists.

    `averages` are the three ratios averaged over the matched pairs;
    when nothing matched, all three are 0.0.
    """

    matched: int
    unmatched: int
    averages: DictRatios


class StructuralScores(BaseModel):
    """Structural similarity scores attached to a pipeline row.

    `matching` is populated only for list-vs-list comparisons; for
    dict-vs-dict it is `None` (there is nothing to pair at the top level).
    """

    matching: float | None = None
    alignment: float
    type_consistency: float
    content_fidelity: float


def compare_dict_structure(d1: Any, d2: Any) -> DictCountTree | LeafMarker:
    """
    Recursively compare two values and produce a raw count tree.

    Walks `d1.keys()` only - the caller (`compute_distance_dicts`)
    invokes this twice with arguments swapped to also see keys present
    only in `d2`. For non-dict inputs, returns a leaf marker that
    classifies the leaf as `"T"` (equal), `"F"` (same type, unequal),
    or `"X"` (different types).

    Parameters
    ----------
    d1, d2 : Any
        Parsed JSON values to compare. Both may be dicts (continues
        recursion) or non-dicts (terminates as a leaf marker).

    Returns
    -------
    DictCountTree | LeafMarker
        A `DictCountTree` if both inputs are dicts, otherwise the leaf
        marker.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        if d1 == d2:
            return "T"
        if type(d1) is not type(d2):
            return "X"
        return "F"

    matched = 0
    unmatched = 0
    nested: list[DictCountTree | LeafMarker] = []

    for key in d1.keys():
        if key not in d2:
            unmatched += 1
            continue
        matched += 1
        nested.append(compare_dict_structure(d1[key], d2[key]))

    return DictCountTree(matched=matched, unmatched=unmatched, nested=nested)


def analyze_comparison(tree: DictCountTree) -> DictCounters:
    """
    Walk a raw count tree and aggregate the six counters.

    `edge_*` accumulate at every level of the dict tree;
    `type_*`/`node_*` accumulate only when the walker encounters a
    leaf marker. Type-mismatched leaves (`"X"`) contribute to
    `type_diff` only - they do not affect `node_*` since value
    equality is not meaningful across different types.

    Parameters
    ----------
    tree : DictCountTree
        Output of `compare_dict_structure` for a dict pair.

    Returns
    -------
    DictCounters
        Aggregated counters.
    """
    edge_equal = tree.matched
    edge_diff = tree.unmatched
    type_equal = type_diff = 0
    node_equal = node_diff = 0

    for entry in tree.nested:
        if entry == "T":
            type_equal += 1
            node_equal += 1
        elif entry == "F":
            type_equal += 1
            node_diff += 1
        elif entry == "X":
            type_diff += 1
        else:
            sub = analyze_comparison(entry)
            edge_equal += sub.edge_equal
            edge_diff += sub.edge_diff
            type_equal += sub.type_equal
            type_diff += sub.type_diff
            node_equal += sub.node_equal
            node_diff += sub.node_diff

    return DictCounters(
        edge_equal=edge_equal,
        edge_diff=edge_diff,
        type_equal=type_equal,
        type_diff=type_diff,
        node_equal=node_equal,
        node_diff=node_diff,
    )


def compute_difference(d1: Any, d2: Any) -> DictCounters:
    """
    Run the dict-pair comparator from `d1`'s perspective.

    Parameters
    ----------
    d1, d2 : Any
        Parsed JSON values; both must be dicts at the top level.

    Returns
    -------
    DictCounters
        Counters from one direction of the comparison.

    Raises
    ------
    ValueError
        If either top-level input is not a dict. The dict pipeline
        cannot consume leaf markers at the top level.
    """
    result = compare_dict_structure(d1, d2)
    if not isinstance(result, DictCountTree):
        raise ValueError("compute_difference requires both inputs to be dicts")
    return analyze_comparison(result)


def compute_distance_dicts(d1: Any, d2: Any) -> DictCounters:
    """
    Compute the symmetric counter set for a dict pair.

    Both directions are run. `edge_diff` is summed across the two
    directions so keys unique to either side count exactly once; all
    other counters are taken from the forward direction after asserting
    they agree with the reversed direction.

    Parameters
    ----------
    d1, d2 : Any
        Top-level dicts to compare.

    Returns
    -------
    DictCounters
        Symmetric counters.

    Raises
    ------
    RuntimeError
        If the forward and reversed counters disagree on a counter that
        should be symmetric. Should not occur on valid inputs - the
        equality of `edge_equal` follows from set-intersection symmetry,
        and `type_*`/`node_*` depend only on shared keys.
    """
    a = compute_difference(d1, d2)
    b = compute_difference(d2, d1)
    if (
        a.edge_equal != b.edge_equal
        or a.type_equal != b.type_equal
        or a.type_diff != b.type_diff
        or a.node_equal != b.node_equal
        or a.node_diff != b.node_diff
    ):
        raise RuntimeError(
            "Forward and reversed dict comparisons disagree on counters "
            "that should be symmetric - this indicates an algorithm bug."
        )
    return DictCounters(
        edge_equal=a.edge_equal,
        edge_diff=a.edge_diff + b.edge_diff,
        type_equal=a.type_equal,
        type_diff=a.type_diff,
        node_equal=a.node_equal,
        node_diff=a.node_diff,
    )


def compute_normalized_distance(d1: dict[str, Any], d2: dict[str, Any]) -> DictRatios:
    """
    Normalize the symmetric counters into three [0, 1] ratios.

    Two empty dicts are treated as vacuously equal — all three ratios
    are 1.0. There is no content on either side to disagree on, and
    this short-circuit avoids a 0/0 in the `alignment` denominator.
    The `type_consistency` and `content_fidelity` denominators are
    guarded with a 0.0 fallback for the disjoint-but-non-empty case
    (no shared keys, so no leaves visited).

    Parameters
    ----------
    d1, d2 : dict[str, Any]
        Top-level dicts to compare.

    Returns
    -------
    DictRatios
        `alignment` is `edge_equal / (edge_equal + edge_diff)`, or 1.0
        when both dicts are empty;
        `type_consistency` is `type_equal / (type_equal + type_diff)`,
        or 0.0 when no leaves were visited;
        `content_fidelity` is `node_equal / (node_equal + node_diff)`,
        or 0.0 when no leaves had matching types.
    """
    if not d1 and not d2:
        return DictRatios(alignment=1.0, type_consistency=1.0, content_fidelity=1.0)
    c = compute_distance_dicts(d1, d2)
    alignment = c.edge_equal / (c.edge_equal + c.edge_diff)
    type_consistency = (
        c.type_equal / (c.type_equal + c.type_diff)
        if (c.type_equal + c.type_diff) > 0
        else 0.0
    )
    content_fidelity = (
        c.node_equal / (c.node_equal + c.node_diff)
        if (c.node_equal + c.node_diff) > 0
        else 0.0
    )
    return DictRatios(
        alignment=alignment,
        type_consistency=type_consistency,
        content_fidelity=content_fidelity,
    )


def compare_list_structure(l1: list[Any], l2: list[Any]) -> ListComparisonResult:
    """
    Pair dicts between two lists by global-greedy on the pair-sum matrix.

    Score every `(i, j)` cell with `DictRatios.total()`, sort cells by
    `(-score, i, j)`, and take disjoint cells whose score strictly
    exceeds `PAIR_THRESHOLD`. Symmetric by construction: the cell set
    and scores are independent of which list drives the iteration, so
    `compare_list_structure(l1, l2)` and `compare_list_structure(l2, l1)`
    select the same pairs (modulo index swap). Duplicate dicts in a
    list are distinguished by their position, since "claimed" is
    tracked by index rather than by value equality.

    Parameters
    ----------
    l1, l2 : list[Any]
        Lists of dicts to pair. Non-dict elements raise.

    Returns
    -------
    ListComparisonResult
        `matched` is the number of pairs taken;
        `unmatched` is `len(l1) - matched` (items in `l1` with no
        partner above threshold; `compute_distance_lists` adds the
        symmetric `len(l2) - matched` for the full count);
        `averages` are the three ratios averaged over the matched
        pairs (all 0.0 when nothing matched).

    Raises
    ------
    ValueError
        If any element of `l1` or `l2` is not a dict; the reference
        does not handle non-dict list elements.
    """
    for item in l1:
        if not isinstance(item, dict):
            raise ValueError("list elements can only be dicts")
    for item in l2:
        if not isinstance(item, dict):
            raise ValueError("list elements can only be dicts")

    cells: list[tuple[float, int, int, DictRatios]] = []
    for i, a in enumerate(l1):
        for j, b in enumerate(l2):
            ratios = compute_normalized_distance(a, b)
            cells.append((ratios.total(), i, j, ratios))
    cells.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_i: set[int] = set()
    used_j: set[int] = set()
    paired: list[DictRatios] = []
    for total, i, j, ratios in cells:
        if total <= PAIR_THRESHOLD:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        paired.append(ratios)

    matched = len(paired)
    unmatched = len(l1) - matched

    if matched > 0:
        averages = DictRatios(
            alignment=sum(r.alignment for r in paired) / matched,
            type_consistency=sum(r.type_consistency for r in paired) / matched,
            content_fidelity=sum(r.content_fidelity for r in paired) / matched,
        )
    else:
        averages = DictRatios(alignment=0.0, type_consistency=0.0, content_fidelity=0.0)

    return ListComparisonResult(matched=matched, unmatched=unmatched, averages=averages)


def compute_distance_lists(l1: list[Any], l2: list[Any]) -> ListComparisonResult:
    """
    Compute the symmetric pairing result for a list pair.

    `compare_list_structure` is symmetric by construction, so `matched`
    and `averages` are direction-independent. The reversed direction's
    unmatched count is just `len(l2) - matched`, derived without a
    second pass through the algorithm.

    Parameters
    ----------
    l1, l2 : list[Any]
        Lists of dicts to pair.

    Returns
    -------
    ListComparisonResult
        `matched` and `averages` from the symmetric pairing;
        `unmatched` is `(len(l1) - matched) + (len(l2) - matched)`,
        the total of items unpaired on either side.
    """
    forward = compare_list_structure(l1, l2)
    return ListComparisonResult(
        matched=forward.matched,
        unmatched=forward.unmatched + (len(l2) - forward.matched),
        averages=forward.averages,
    )


def compute_list_of_dict_normalized_distance(
    l1: list[Any], l2: list[Any]
) -> StructuralScores:
    """
    Top-level numeric output for two lists.

    Two empty lists are treated as vacuously equal — all four scores
    are 1.0. There is no content on either side to disagree on, and
    this short-circuit avoids a 0/0 in the `matching` denominator.

    Parameters
    ----------
    l1, l2 : list[Any]
        Lists of dicts to compare.

    Returns
    -------
    StructuralScores
        With `matching = matched / (matched + unmatched)` populated and
        the three averaged ratios filled in. For the empty-vs-empty
        case, all four fields are 1.0.
    """
    if not l1 and not l2:
        return StructuralScores(
            matching=1.0,
            alignment=1.0,
            type_consistency=1.0,
            content_fidelity=1.0,
        )
    r = compute_distance_lists(l1, l2)
    matching = r.matched / (r.matched + r.unmatched)
    return StructuralScores(
        matching=matching,
        alignment=r.averages.alignment,
        type_consistency=r.averages.type_consistency,
        content_fidelity=r.averages.content_fidelity,
    )


def structural_scores(j1: Any, j2: Any) -> StructuralScores | None:
    """
    Compute structural similarity between two parsed JSON values.

    Dispatches on the top-level shape of `j1` and `j2`: dict-vs-dict
    yields scores without `matching`; list-vs-list yields all four
    scores.

    Parameters
    ----------
    j1, j2 : Any
        Parsed JSON values to compare. The pipeline only feeds
        schema-validated outputs (always dict for ODRL, list for
        DCPL); the `None` fallback is purely defensive.

    Returns
    -------
    StructuralScores | None
        Populated `StructuralScores` for valid inputs (including empty
        list-vs-list, which collapses to all-1.0); `None` for
        mismatched top-level shapes or non-JSON inputs.
    """
    if isinstance(j1, dict) and isinstance(j2, dict):
        r = compute_normalized_distance(j1, j2)
        return StructuralScores(
            alignment=r.alignment,
            type_consistency=r.type_consistency,
            content_fidelity=r.content_fidelity,
        )
    if isinstance(j1, list) and isinstance(j2, list):
        return compute_list_of_dict_normalized_distance(j1, j2)
    return None
