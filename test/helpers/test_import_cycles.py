import importlib


def test_helpers_and_adapter_can_import_together():
    helpers = importlib.import_module("mellea.helpers")
    adapter = importlib.import_module("mellea.backends.adapters.adapter")

    assert helpers._ServerType is adapter._ServerType
