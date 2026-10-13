from pathlib import Path


def qualifyname(directoryname, filename=None):
    directory = Path(directoryname)
    if filename is None:
        return directory
    return directory / filename


def code(filename=None):
    return qualifyname(Path(__file__).parent.resolve(), filename)


def base(filename=None):
    return qualifyname(code().parent, filename)


def tests(filename=None):
    return qualifyname(code("tests"), filename)
