import pytest
from services import service_registry
sm = service_registry.get_station_manager()

@pytest.mark.parametrize(
    "name,sids",
    [
        ("府中", ["BL06", "081"]),   # 取用你知道的 code / sid
        ("動物園", ["BR01", "019"]),
        ("台北車站", ["BL12", "R10", "051"]),  # 可多筆
    ],
)
def test_get_station_ids(name, sids):
    ids = sm.get_station_ids(name)
    # 只要至少對到一筆即可
    assert any(x in ids for x in sids)

