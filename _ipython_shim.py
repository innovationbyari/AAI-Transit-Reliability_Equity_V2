"""Stand-ins for IPython.display used when the notebook code runs as a plain Python module (Streamlit).

The notebook calls display(...) to show tables, widgets and banners. Outside a Jupyter kernel there is nowhere to show
them, so display() does nothing and HTML / Markdown simply hold their text.
"""


def display(*objs, **kwargs):
    return None


def clear_output(*args, **kwargs):
    return None


class _Rich:
    def __init__(self, data=None, *args, **kwargs):
        self.data = data

    def __repr__(self):
        return str(self.data)

    def _repr_markdown_(self):
        return str(self.data)


class HTML(_Rich):
    pass


class Markdown(_Rich):
    pass


class Image(_Rich):
    pass
