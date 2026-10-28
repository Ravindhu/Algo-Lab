"""MT5 adapter contract tests. They run on any OS: the MetaTrader5 package is replaced by a fake
module. The live test runs only on Windows with a logged-in terminal."""
import asyncio
import sys
import types

import pytest

from algolab.live import adapter as A


def fake_mt5(**overrides):
    """A module with every required name, so missing_api() reports nothing."""
    mod = types.ModuleType("MetaTrader5")
    for n in A.REQUIRED_FUNCTIONS:
        setattr(mod, n, lambda *a, **k: None)
    for n in A.REQUIRED_CONSTANTS:
        setattr(mod, n, 1)
    for t in A.TIMEFRAMES:
        setattr(mod, f"TIMEFRAME_{t}", 1)
    for k, v in overrides.items():
        setattr(mod, k, v)
    return mod


def test_complete_module_has_no_missing_api():
    assert A.missing_api(fake_mt5()) == []


def test_missing_function_is_reported():
    mod = fake_mt5()
    del mod.copy_rates_from_pos
    assert "copy_rates_from_pos" in A.missing_api(mod)


def test_missing_timeframe_is_reported():
    mod = fake_mt5()
    del mod.TIMEFRAME_H4
    assert "TIMEFRAME_H4" in A.missing_api(mod)


def test_self_test_flags_problems_and_blocks_trading():
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5()
    del ad.mt5.order_send
    problems, _ = asyncio.run(ad.self_test())
    assert any("order_send" in p for p in problems)


def test_open_positions_none_means_unknown_not_zero():
    """A failed positions_get must fail closed (None), never look like 'no positions'."""
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(positions_get=lambda *a, **k: None)
    assert asyncio.run(ad.open_positions()) is None
    ad.mt5 = fake_mt5(positions_get=lambda *a, **k: ())
    assert asyncio.run(ad.open_positions()) == 0


def test_field_reads_object_and_dict():
    class Obj:
        equity = 100.0
    assert A._field(Obj(), "equity") == 100.0
    assert A._field({"equity": 5}, "equity") == 5
    assert A._field({}, "equity", default=0) == 0


@pytest.mark.skipif(sys.platform != "win32", reason="needs the MetaTrader 5 terminal")
def test_live_contract():
    async def go():
        async with A.Mt5Adapter() as ad:
            problems, _ = await ad.self_test()
            assert not problems, problems
    asyncio.run(go())
