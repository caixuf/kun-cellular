"""四展示演示后端契约测试。

断言 frontend/{immune,ecosystem,locomotion,slingshot}.html 所消费的
/api/*/status JSON 具有精确的字段与类型 —— 这正是曾经漂移、导致页面半坏的契约。

在临时端口(0)启动后端, 不干扰 8333/8833 上可能运行的实例。
"""
import json
import threading
import urllib.request

import pytest

be = pytest.importorskip("tools.cellular_live_backend")


@pytest.fixture(scope="module")
def base_url():
    srv = be.ThreadedHTTPServer(("127.0.0.1", 0), be.ObservatoryHTTPHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _assert_keys(obj, keys, where):
    missing = [k for k in keys if k not in obj]
    assert not missing, f"{where} 缺少字段: {missing}; 实际={sorted(obj.keys())}"


def _assert_element_keys(arr, keys, where):
    assert isinstance(arr, list), f"{where} 应为 list"
    for i, el in enumerate(arr):
        assert isinstance(el, dict), f"{where}[{i}] 应为对象"
        _assert_keys(el, keys, f"{where}[{i}]")


def test_immune_status_schema(base_url):
    d = _get(base_url, "/api/immune/status")
    _assert_keys(d, ["generation", "step_count", "max_steps", "clearance_rate",
                     "pathogens_alive", "total_pathogens", "history_clearance",
                     "pathogens", "macrophages"], "immune")
    assert isinstance(d["clearance_rate"], (int, float))
    assert isinstance(d["history_clearance"], list)
    _assert_element_keys(d["pathogens"], ["id", "x", "y", "alive", "type"], "immune.pathogens")
    _assert_element_keys(d["macrophages"], ["id", "x", "y", "radius", "chem_r"], "immune.macrophages")
    if d.get("real"):
        assert len(d["macrophages"]) > 0, "real 模式应输出巨噬细胞显示层"


def test_ecosystem_status_schema(base_url):
    d = _get(base_url, "/api/eco/status")
    _assert_keys(d, ["generation", "step_count", "max_steps", "prey_alive", "total_prey",
                     "total_hunts", "history_prey", "history_pred", "food", "prey", "predators"],
                 "eco")
    _assert_element_keys(d["food"], ["x", "y"], "eco.food")
    _assert_element_keys(d["prey"], ["x", "y", "theta", "alive"], "eco.prey")
    _assert_element_keys(d["predators"], ["x", "y", "theta"], "eco.predators")
    assert isinstance(d["history_prey"], list) and isinstance(d["history_pred"], list)


def test_locomotion_status_schema(base_url):
    d = _get(base_url, "/api/loco/status")
    _assert_keys(d, ["generation", "step_count", "max_steps", "best_distance",
                     "history_dist", "champion"], "loco")
    champ = d["champion"]
    _assert_keys(champ, ["nodes", "muscles"], "loco.champion")
    _assert_element_keys(champ["nodes"], ["x", "y"], "loco.champion.nodes")
    _assert_element_keys(champ["muscles"], ["n1", "n2"], "loco.champion.muscles")
    assert isinstance(d["history_dist"], list)


def test_slingshot_status_schema(base_url):
    d = _get(base_url, "/api/slingshot/status")
    _assert_keys(d, ["generation", "step_count", "max_steps", "success_rate",
                     "history_success", "target", "stars", "probes"], "slingshot")
    _assert_keys(d["target"], ["x", "y", "r"], "slingshot.target")
    _assert_element_keys(d["stars"], ["x", "y", "color"], "slingshot.stars")
    _assert_element_keys(d["probes"], ["id", "x", "y", "alive", "reached", "trail"],
                         "slingshot.probes")
    assert isinstance(d["history_success"], list)
    for p in d["probes"]:
        assert isinstance(p["trail"], list)
        for pt in p["trail"]:
            assert len(pt) == 2
