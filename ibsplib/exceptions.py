__all__ = ["StructureSizeNotDefined", "LumpIndexOutOfBounds", "ErrorReadingBSPFile", "NotIBSPFileException", "FileDoesNotExist",
           "UnsupportedBSPFileVersion"]


class StructureSizeNotDefined(Exception):
    pass


class LumpIndexOutOfBounds(Exception):
    pass


class ErrorReadingBSPFile(Exception):
    pass


class NotIBSPFileException(Exception):
    pass


class FileDoesNotExist(Exception):
    pass


class UnsupportedBSPFileVersion(Exception):
    pass
