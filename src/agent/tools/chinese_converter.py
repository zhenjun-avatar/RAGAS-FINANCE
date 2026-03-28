import opencc


class ChineseConverter:
    def __init__(self):
        self.s2t_converter = opencc.OpenCC("s2t")
        self.t2s_converter = opencc.OpenCC("t2s")

    def convert(self, text: str, mode: str = "s2t") -> str:
        if not text:
            return text
        if mode == "s2t":
            return self.s2t_converter.convert(text)
        if mode == "t2s":
            return self.t2s_converter.convert(text)
        raise ValueError(f"Unsupported mode: {mode}. Use 's2t' or 't2s'.")


_converter: ChineseConverter | None = None


def get_converter() -> ChineseConverter:
    global _converter
    if _converter is None:
        _converter = ChineseConverter()
    return _converter


def convert_text(text: str, mode: str) -> str:
    return get_converter().convert(text, mode)
