import regex as re


def pad_punctuation(text):
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r"([^_\s\p{N}\p{L}\p{M}])", r" \1 ", str(text))
    # Collapse consecutive whitespace into one space.
    text = re.sub(r"\s+", " ", text)
    return text
