class BaseData:
    def scale(self, x):
        raise UserWarning("scale method not implemented for dataset")

    def inverse_scale(self, x):
        raise UserWarning("inverse_scale not implemented for dataset")

    @classmethod
    def add_cli(self, parser):
        raise UserWarning("add_cli not implemented for dataset class")

