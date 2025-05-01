
def get_function(name):
    if name == "parabola":
        from .parabola import f, df
    elif name == "polynomial":
        from .polynomial import f, df
    elif name == "sine":
        from .sine import f, df
    else:
        raise ValueError("Unknown function name")
    return f, df
