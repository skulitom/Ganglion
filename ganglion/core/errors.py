"""The one error the runtime raises at its public edge: a code the bridge can forward, and a message."""


class RuntimeErrorWithCode(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
