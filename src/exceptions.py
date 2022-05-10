# src/exceptions


class InputError(Exception):
    def __init__(self, operator_name, msg):
        super().__init__()
        self.msg = "Unable to identify operator '{0}'. {1}".format(operator_name, msg)

    def __str__(self):
        return self.msg
