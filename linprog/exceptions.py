class BasisIsPrimalInfeasibleError(Exception):
    pass


class BasisIsDualInfeasibleError(Exception):
    pass


class PrimalIsUnboundedError(Exception):
    pass


class PrimalIsInfeasibleError(Exception):
    pass


class DualIsUnboundedError(Exception):
    pass


class DualIsInfeasibleError(Exception):
    pass
