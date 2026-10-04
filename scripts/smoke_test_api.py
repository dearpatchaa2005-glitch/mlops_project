import json
import urllib.error
import urllib.request


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


good_row = {
    "age_approx": 55, "sex": "female", "anatom_site_general": "upper extremity",
    "clin_size_long_diam_mm": 4.2, "tbp_tile_type": "3D: XP",
    "tbp_lv_A": 10.1, "tbp_lv_Aext": 9.8, "tbp_lv_B": 12.3, "tbp_lv_Bext": 11.9,
    "tbp_lv_C": 15.0, "tbp_lv_Cext": 14.7, "tbp_lv_H": 20.0, "tbp_lv_Hext": 19.5,
    "tbp_lv_L": 50.0, "tbp_lv_Lext": 49.2, "tbp_lv_areaMM2": 12.5,
    "tbp_lv_area_perim_ratio": 3.1, "tbp_lv_color_std_mean": 1.2,
    "tbp_lv_deltaA": 0.3, "tbp_lv_deltaB": -0.2, "tbp_lv_deltaL": 0.8,
    "tbp_lv_deltaLB": 1.1, "tbp_lv_deltaLBnorm": 2.2, "tbp_lv_eccentricity": 0.6,
    "tbp_lv_minorAxisMM": 2.0, "tbp_lv_norm_border": 1.5, "tbp_lv_norm_color": 1.8,
    "tbp_lv_perimeterMM": 14.0, "tbp_lv_radial_color_std_max": 0.9,
    "tbp_lv_stdL": 2.3, "tbp_lv_stdLExt": 2.1, "tbp_lv_symm_2axis": 0.7,
    "tbp_lv_symm_2axis_angle": 45.0, "tbp_lv_x": 10.0, "tbp_lv_y": -5.0, "tbp_lv_z": 2.0,
}

status, body = post("/predict", {"instances": [good_row]})
print("good row ->", status, body.get("predictions"))

bad_row = dict(good_row)
bad_row["sex"] = "alien"
bad_row["tbp_lv_eccentricity"] = 99.0
status, body = post("/predict", {"instances": [bad_row]})
print("bad category/range ->", status, json.dumps(body)[:300])

missing_field_row = dict(good_row)
del missing_field_row["tbp_lv_x"]
status, body = post("/predict", {"instances": [missing_field_row]})
print("missing nullable field ->", status, json.dumps(body)[:300])

non_float_row = dict(good_row)
non_float_row["age_approx"] = "not-a-number"
status, body = post("/predict", {"instances": [non_float_row]})
print("non-numeric value (pydantic layer) ->", status, json.dumps(body)[:200])

extra_field_row = dict(good_row)
extra_field_row["unexpected_field"] = 1
status, body = post("/predict", {"instances": [extra_field_row]})
print("unexpected extra field ->", status, json.dumps(body)[:200])
