"""String distance metrics used as similarity signals."""


def hamming_distance(s1: str, s2: str) -> int:
    """
    Compute the Hamming distance between two equal-length strings.

    Parameters
    ----------
    s1, s2 : str
        Input strings. Must have the same length.

    Returns
    -------
    int
        The number of positions at which the two strings differ.

    Raises
    ------
    ValueError
        If `s1` and `s2` have different lengths, since the Hamming
        distance is only defined for equal-length sequences.
    """
    if len(s1) != len(s2):
        raise ValueError(
            f"Hamming distance requires equal-length inputs, "
            f"got {len(s1)} and {len(s2)}"
        )
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    insertions, deletions, or substitutions required to transform `s1`
    into `s2`. The implementation uses the classic two-row dynamic
    programming formulation, indexing the inner row by the shorter
    string to keep memory at O(min(len(s1), len(s2))).

    Parameters
    ----------
    s1, s2 : str
        Input strings. May have different lengths.

    Returns
    -------
    int
        The edit distance between `s1` and `s2`.
    """
    # Swap so that s1 is the longer string; the inner DP row then scales
    # with the shorter string, halving memory in the worst case.
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: list[int] = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + int(c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
