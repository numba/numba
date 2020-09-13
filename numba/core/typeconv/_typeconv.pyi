import typing as pt


class _TypeManager:
    ...


def new_type_manager() -> _TypeManager:
    ...


def select_overload(
    tmcap: _TypeManager,
    sigtup: pt.Sequence[int],
    ovsigstup: pt.Sequence[pt.Sequence[int]],
    allow_unsafe: bool,
    exact_match_required: bool,
) -> int:
    ...


def check_compatible(
    tmcap: _TypeManager,
    from_: int,
    to: int,
) -> pt.Optional[str]:
    ...


def set_compatible(
    tmcap: _TypeManager,
    from_: int,
    to: int,
    by: int,
) -> None:
    ...


def get_pointer(
    tmcap: _TypeManager,
) -> int:
    ...
