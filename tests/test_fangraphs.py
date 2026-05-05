from statbirt.fangraphs import _stuff_value_from_row


def test_stuff_value_accepts_fangraphs_api_field():
    assert _stuff_value_from_row({"sp_stuff": "101.7"}) == 101.7
