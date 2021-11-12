flags = [
    "-Wall",
    "-Wextra",
    "-Werror"
    "-I", "/usr/local/cuda/include",
]


def FlagsForFile(filename):
    return {
        "flags": flags
    }
