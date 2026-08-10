# Flatfielded model comparison (v1 / v2 / v3)

Predicted crack area fraction. Undamaged specimens SHOULD read near zero;
damaged ones should still find their crack; B2_338_13 has ~29.7% ground truth.

| image | v1 (16 img, 50% crack) | v2 (71 img, 27.5% crack) | v3 (71 img, 57.5% crack, neg-capped) |
|---|---|---|---|
| wrought_0cyc UNDAMAGED | 37.7% | 15.1% | **7.1%** |
| b3_amb UNDAMAGED | 26.7% | 5.2% | **3.5%** |
| B2_amb_2 UNDAMAGED | 26.9% | 40.1% | **7.4%** |
| wrought_1300 damaged | 20.3% | 13.8% | 16.3% |
| AM_1000 damaged | 18.3% | 11.7% | 9.1% |
| B2_338_13 training (GT ~29.7%) | 30.2% | 36.1% | **32.9%** |

v3 wins on every row that matters. It also RESOLVES the v2 regression whose
mechanism was initially only partially understood: capping the negative-only
corrections (--neg-cap 3000) took B2_amb_2 from 40.1% back to 7.4%, confirming
the class-balance shift (50% -> 27.5% crack, causing class_weight='balanced'
to upweight crack ~2.6x) was indeed the cause.

## v3 is NOT deployed, and must not be dropped into the raw-input path

v3 expects FLATFIELDED input. The paint tool and apply_pixel_model.py feed RAW
images to models/pixel_hgb_final.joblib. Copying v3 there would silently create
an input-distribution mismatch (flatfielded-trained model fed raw pixels) and
produce worse results than either model alone. Deploying v3 requires switching
the inference/paint path over to flatfielded input as well -- a deliberate,
separate change.

## Remaining gap

All 760k crack training pixels still come from B2. The 59 new images contributed
only force-not-crack labels, so v3 has seen zero positive crack examples from AM,
Wrought, or B3 microstructure. It is validated for "stop flooding non-specimen
area", not for those materials' crack morphology.
