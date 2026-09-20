# Adapted from KoelLabs/ML scripts/data_loaders/SpeechAccent.py (AGPL-3.0).
WORD_LIST = {'please', 'call', 'stella', 'stellaw', 'ask', 'her', 'to', 'bring', 'these', 'things', 'with', 'from', 'the', 'store', 'six', 'spoons', 'of', 'fresh', 'snowpeas', 'five', 'thick', 'slabs', 'blue', 'cheese', 'and', 'maybe', 'a', 'snack', 'for', 'brother', 'bob', 'we', 'also', 'need', 'small', 'plastic', 'snake', 'big', 'toy', 'frog', 'for', 'kids', 'she', 'can', 'scoop', 'things', 'into', 'three', 'red', 'bags', 'we', 'will', 'go', 'meet', 'her', 'wednesday', 'at', 'train', 'station', 'stationx', '-del', 'sp'}  # fmt: skip


def split_words_with_dictionary(text, word_list=WORD_LIST):
    n = len(text)
    dp = [False] * (n + 1)
    dp[0] = True
    result = [[]] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j]:
                word = text[j:i]
                if word.lower() in word_list:
                    dp[i] = True
                    if j == 0:
                        result[i] = [word]
                    else:
                        result[i] = result[j] + [word]
                    break

    assert dp[n], text
    return result[n] if dp[n] else text


def clean_text(text):
    text = text.replace("`", "")
    text = (
        text.replace("-noise", "")
        .replace("-hes", "")
        .replace("-insrt", "")
        .replace("-laugh", "")
        .replace("-sub", "")
        .replace("-missing", "")
        .replace("-birdsinging", "")
        .replace("-cough", "")
        .replace("-check-as", "")
        .replace("-check", "")
        .replace("-change", "")
        .replace("-experimenterspeechinbackground", "")
        .replace("-missound", "")
        .replace("-rep-del", "")
        .replace("-jes", "")
        .replace("1", "")
        .replace("_s", "")
        .replace("-regroup", "")
        .replace("-kof", "")
    )
    text = text.replace("2", "to")
    parts = (
        ((t.replace("-rep", ""), t.replace("-rep", "")) if t.endswith("-rep") else (t,))
        for t in text.split(" ")
    )
    parts = (t.removesuffix("-").removesuffix("-rep") for p in parts for t in p)
    parts = [w for t in parts for w in split_words_with_dictionary(t) if w != "sp"]
    while "-del" in parts:
        ix = parts.index("-del")
        parts = parts[: ix - 1] + parts[ix + 1 :]
    parts = ("stella" if p == "stellaw" else p for p in parts)
    parts = ("station" if p == "stationx" else p for p in parts)
    text = " ".join(parts)
    return text
