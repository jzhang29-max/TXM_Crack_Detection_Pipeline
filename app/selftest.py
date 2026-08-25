"""
End-to-end self test. Exercises every user-facing feature against a running
server and reports pass/fail per feature.

    python3 app/server.py &            # or ./run_app.sh
    python3 app/selftest.py            # default http://127.0.0.1:8800
    python3 app/selftest.py --base http://127.0.0.1:3000 --retrain

Ships with the package so a new user can verify their install rather than trust
a README. Everything except --retrain runs in about a minute; --retrain adds a
few minutes because it trains and then validates against the ground truth.

It creates its own test image from the built-in ground truth, so it needs no
data of your own, and it cleans up after itself unless --keep is given.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name}  -- {why}", flush=True)


def req(base, path, method="GET", body=None, files=None, timeout=600):
    url = base.rstrip("/") + path
    if files:
        boundary = "----txmselftest"
        parts = []
        for field, (fname, content) in files.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f'name="{field}"; filename="{fname}"\r\n'
                         f"Content-Type: application/octet-stream\r\n\r\n".encode())
            parts.append(content)
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        r = urllib.request.Request(url, data=data, method="POST")
        r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(url, data=data, method=method)
        if data:
            r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, (json.loads(raw) if "json" in ctype else raw)
    except urllib.error.HTTPError as e:
        # A 4xx is a legitimate answer to test -- "this file is not readable" is
        # supposed to be a 400 with a JSON explanation. urlopen raises on those and
        # str(HTTPError) does not include the body, so read it here and return it
        # like any other response; callers already inspect .ok / .error.
        raw = e.read()
        ctype = e.headers.get("Content-Type", "") if e.headers else ""
        try:
            return e.code, (json.loads(raw) if "json" in ctype else raw)
        except Exception:
            return e.code, raw


def S_label(resp):
    return ((resp or {}).get("current") or {}).get("label")


def wait_job(base, jid, label, timeout=5400):
    """Poll a job to completion.

    5400 s, not 2400: a retrain now gathers features from every labelled image, and after
    the research negatives were imported that is 71 images rather than 2. Add the
    crack-free false-positive check and the whole pass runs ~30 min, so the old 40-minute
    cap could expire on a HEALTHY retrain. It also returned dict(state="timeout") with no
    "seconds" key, so the failure printed as "Nones" and read like a crash -- it now says
    plainly that it gave up waiting, which is a different problem from a job that died.
    """
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        _, j = req(base, f"/api/job/{jid}")
        if j.get("stage") != last:
            last = j.get("stage")
            print(f"        {label}: {last}", flush=True)
        if j["state"] != "running":
            return j
        time.sleep(4)
    return dict(state="timeout", seconds=timeout,
                error=f"still running after {timeout}s -- the self test gave up waiting; "
                      f"the job itself may still finish. Check /api/jobs.")


def make_test_tiff():
    """A small test image cropped from one of the shipped TXM images.

    It used to come from dataset_cache's expanded ground-truth arrays. Those are gone --
    they were the externally-labelled masks and the images that went with them -- so it now
    crops a corner out of the smallest file in images/, which is the raw data this project
    is actually about. Cropped rather than whole so ingest stays quick.
    """
    import tifffile
    imgdir = os.path.join(PROJECT, "images")
    if not os.path.isdir(imgdir):
        return None, None
    cands = [f for f in os.listdir(imgdir) if f.lower().endswith((".tif", ".tiff"))]
    if not cands:
        return None, None
    cands.sort(key=lambda f: os.path.getsize(os.path.join(imgdir, f)))
    a = tifffile.imread(os.path.join(imgdir, cands[0]))
    # 1536 square, not a smaller crop: the paint tests stroke at coordinates up to
    # x=1320, y=1000 with radii up to 50, and the old dataset_cache image happened to be
    # ~1700 px so they fitted. A 768 crop silently put four of them outside the image, and
    # the failures read like paint and undo bugs rather than an out-of-bounds test fixture.
    a = np.asarray(a)[:1536, :1536].astype(np.float64)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    buf = io.BytesIO()
    tifffile.imwrite(buf, (a * 65535).astype(np.uint16))
    return "SELFTEST_IMAGE.tif", buf.getvalue()


def check_model_picker(B, focus=None):
    """Switching to a model already computed for an image must be instant, not a re-predict.

    CALLED BEFORE THIS TEST UPLOADS ANYTHING, and that is the whole point. A model other
    than the current one can never hold a prediction for an image this run just created,
    so while this check lived after the upload it required something impossible and
    skipped on every single run -- coverage that read as present and was not. That is the
    same blind spot the dedented-P.ingest bug hid in, so it is worth keeping honest.

    Run before the uploads, a fully cached model does exist, and the strict assertion
    holds: neither trip may queue a prediction job.
    """
    try:
        _, ml = req(B, "/api/models", timeout=120)
        models = ml.get("models") or []
        cur = next((m for m in models if m.get("current")), None)
        check("model list offers the current model", cur is not None,
              f"{len(models)} model(s): " + ", ".join(m["label"] for m in models[:3]))
        if cur is None:
            return

        # Every model must carry its MEASURED false-positive rate on the crack-free
        # specimens, or an explicit null. The picker showed nothing but a name until a
        # user switched to a model that marks 22% of blank specimen as crack and had no
        # way to find that out from the interface.
        measured = [m for m in models if m.get("clean_fp") is not None]
        check("model list reports measured background error",
              all("clean_fp" in m and "fp_warn" in m for m in models),
              f"{len(measured)}/{len(models)} measurable: " +
              ", ".join(f"{m['label'].split()[-1]}={m['clean_fp']*100:.2f}%" for m in measured[:4]))
        if len(measured) >= 2:
            best = min(m["clean_fp"] for m in measured)
            expect = {m["id"]: m["clean_fp"] > max(best * 5, best + 0.01) for m in measured}
            check("a model much worse on crack-free specimen is flagged",
                  all(m["fp_warn"] == expect[m["id"]] for m in measured),
                  f"flagged {sum(1 for m in measured if m['fp_warn'])} of {len(measured)}, "
                  f"best {best*100:.2f}%")
            worst = max(measured, key=lambda m: m["clean_fp"])
            if worst["clean_fp"] > max(best * 5, best + 0.01):
                check("the worst measured model is the one flagged", worst["fp_warn"],
                      f"{worst['label']} at {worst['clean_fp']*100:.1f}%")
        # Only switch to a model ALREADY cached for every loaded image. Switching to an
        # uncached one queues a real prediction pass -- correct product behaviour, awful
        # test: with 71 images that once spent 48 minutes re-predicting the library and
        # left the app on another model while it did.
        ready = [m for m in models
                 if not m.get("current") and m.get("cached_for") == m.get("n_images")]
        others = [m for m in models if not m.get("current")]
        if not others:
            skip("switching models is instant when cached",
                 "only one model exists until you retrain")
        elif not ready:
            skip("switching models is instant when cached",
                 f"{len(others)} other model(s) exist but none is computed for all "
                 f"{cur.get('n_images')} image(s); switching would queue a real "
                 f"prediction pass rather than exercise the cache")
        else:
            t0 = time.time()
            body = dict(id=ready[0]["id"])
            if focus:
                body["focus"] = focus
            _, away = req(B, "/api/model/select", "POST", body, timeout=120)
            _, back = req(B, "/api/model/select", "POST", dict(id=cur["id"]), timeout=120)
            dt = time.time() - t0
            check("switching models is instant when cached",
                  away.get("ok") is True and not away.get("job")
                  and back.get("ok") is True and not back.get("job"),
                  f"round trip in {dt:.2f}s with no prediction job, "
                  f"{len(back.get('instant') or [])} image(s) from cache")
            check("switching back restores the original model",
                  S_label(back) == cur["label"], f"{S_label(back)}")
    except Exception as e:                                      # noqa: BLE001
        check("model picker", False, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8800")
    ap.add_argument("--retrain", action="store_true", help="also test retrain (slow)")
    ap.add_argument("--keep", action="store_true", help="do not delete the test image")
    args = ap.parse_args()
    B = args.base
    print(f"TXM app self test against {B}\n")

    # ---- server + model
    try:
        _, m = req(B, "/api/model", timeout=30)
    except Exception as e:
        print(f"  FAIL  server reachable -- {type(e).__name__}: {e}")
        print("\n  Is it running?  python3 app/server.py")
        sys.exit(1)
    check("server reachable", True)
    check("model loaded", "NOT LOADED" not in m.get("description", ""), m.get("description", ""))
    # THE GATE'S REAL PREREQUISITES, since there is no external ground truth any more.
    # This used to assert `ground_truth_available`, a flag the server stopped reporting when
    # the externally-labelled masks were deleted. What a retrain actually needs is labelled
    # images to cross-validate over (at least two, so a fold can hold one out) and at least
    # one confirmed crack-free specimen to measure false positives on.
    try:
        sys.path.insert(0, os.path.join(PROJECT, "app", "core"))
        sys.path.insert(0, os.path.join(PROJECT, "code"))
        import pipeline as _Pg, store as _Sg
        _lab = [x for x in _Sg.list_images()
                if x.get("corrected_crack_px") or x.get("corrected_not_px")]
        _clean = [x for x in _Sg.list_images()
                  if any(k.lower() in (x.get("filename") or "").lower()
                         for k in _Pg.CLEAN_SPECIMENS)]
        check("a retrain has something to validate against",
              len(_lab) >= 2 and len(_clean) >= 1,
              f"{len(_lab)} labelled image(s) for the grouped cross-validation, "
              f"{len(_clean)} confirmed crack-free specimen(s) for the false-positive axis")
    except Exception as e:                                      # noqa: BLE001
        check("a retrain has something to validate against", False, str(e))

    # Before uploading anything: the picker check needs a model cached for every image,
    # which stops being true the moment this run adds one.
    _, pre = req(B, "/api/images")
    check_model_picker(B, focus=(pre["images"][0]["id"] if pre.get("images") else None))

    # ---- upload + ingest
    fname, content = make_test_tiff()
    if content is None:
        print("  FAIL  could not build a test image (images/ missing or empty)")
        sys.exit(1)
    _, up = req(B, "/api/upload", files={"files": (fname, content)}, timeout=120)
    check("upload accepted", up.get("ok") is True)
    iid = (up.get("added") or up.get("reused") or [None])[0]
    check("upload returned an image id", bool(iid), str(iid))
    if up.get("job"):
        j = wait_job(B, up["job"], "ingest")
        check("ingest completed (preprocess + SAM + predict)", j["state"] == "done",
              j.get("error") or f"{j.get('seconds')}s")

    _, imgs = req(B, "/api/images")
    me = next((x for x in imgs["images"] if x["id"] == iid), {})
    check("image marked ready", me.get("status") == "ready", me.get("status", "?"))
    check("prediction produced", me.get("predicted_area") is not None,
          f"{(me.get('predicted_area') or 0)*100:.2f}% crack")
    check("display image is preprocessed", "destitch" in str(me.get("display", "")),
          str(me.get("display")))

    # ---- file formats the UI advertises must actually load.
    # The drop zone offers .tif .tiff .png and the file picker adds .jpg, but ingest
    # read everything with tifffile.imread, so a dropped PNG was accepted, stored,
    # and then died with "not a TIFF file: header=b'\x89PNG'". Tested here because
    # it shipped: the previous suite only ever uploaded a TIFF.
    try:
        from PIL import Image as _I
        buf = io.BytesIO()
        _I.fromarray((np.linspace(0, 255, 256 * 256).reshape(256, 256)).astype(np.uint8)).save(buf, format="PNG")
        _, pu = req(B, "/api/upload", files={"files": ("SELFTEST_PNG.png", buf.getvalue())}, timeout=120)
        pid = (pu.get("added") or pu.get("reused") or [None])[0]
        pj = wait_job(B, pu["job"], "png ingest") if pu.get("job") else {"state": "done"}
        check("PNG upload ingests (not just TIFF)", pu.get("ok") is True and pj["state"] == "done",
              pj.get("error") or "ok")
        if pid:
            req(B, f"/api/image/{pid}", "DELETE")
    except Exception as e:
        check("PNG upload ingests (not just TIFF)", False, str(e))

    # An unreadable file should be refused at upload with a sentence, not stored and
    # then failed inside a background job with a decoder traceback.
    try:
        st_rej, rj = req(B, "/api/upload", files={"files": ("SELFTEST_NOTES.txt", b"not an image")}, timeout=60)
        check("unreadable file is refused with a clear message",
              rj.get("ok") is False and "supported" in (rj.get("error") or ""),
              (rj.get("error") or "")[:70])
    except Exception as e:
        # req() raises on a 400, which is itself the correct behaviour -- accept it
        # as long as the message names what is supported.
        check("unreadable file is refused with a clear message", "supported" in str(e), str(e)[:70])

    # ---- rendering endpoints
    for path, label in [(f"/api/image/{iid}/display.png", "display PNG renders"),
                        (f"/api/image/{iid}/mask.png", "overlay PNG renders")]:
        try:
            st, raw = req(B, path, timeout=120)
            check(label, st == 200 and raw[:4] == b"\x89PNG", f"{len(raw)} bytes")
        except Exception as e:
            check(label, False, str(e))

    # ---- painting.
    # Start from a known state. Re-uploading the same file reuses its entry (the id
    # is a content hash), so an earlier run -- or an earlier ABORTED run -- can leave
    # corrections behind. Asserting absolute pixel counts then fails for a reason
    # that has nothing to do with the app, which is exactly what happened once.
    req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
    _, base = req(B, f"/api/image/{iid}/stats")
    _, r = req(B, f"/api/image/{iid}/correction", "POST",
               dict(mode="crack", radius=30, points=[[400, 400], [440, 410], [480, 420]]))
    check("paint crack stroke", r.get("crack_px", 0) > 0, f"{r.get('crack_px'):,} px")
    crack_after_first = r.get("crack_px")
    not_after_first = r.get("not_px")
    d1 = r.get("undo_depth")
    _, r2 = req(B, f"/api/image/{iid}/correction", "POST",
                dict(mode="erase", radius=30, points=[[900, 500], [940, 510]]))
    check("paint eraser stroke", r2.get("not_px", 0) > not_after_first,
          f"{r2.get('not_px'):,} px")
    check("undo stack grows per stroke", r2.get("undo_depth") == d1 + 1,
          f"{d1} -> {r2.get('undo_depth')}")

    # ---- undo: must restore exactly the state before the LAST stroke
    _, u = req(B, f"/api/image/{iid}/undo", "POST")
    check("undo removes only the last stroke",
          bool(u.get("ok")) and u.get("not_px") == not_after_first
          and u.get("crack_px") == crack_after_first,
          f"crack {u.get('crack_px'):,} (want {crack_after_first:,}) / "
          f"not {u.get('not_px'):,} (want {not_after_first:,})")

    # ---- region removal. Probe a point that is actually INSIDE a predicted
    # region, found from the mask itself -- a hard-coded coordinate silently
    # skipped this test whenever it landed on background.
    probe = (850, 850)
    try:
        from PIL import Image
        from scipy import ndimage as _ndi
        _, raw = req(B, f"/api/image/{iid}/mask.png", timeout=120)
        alpha = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))[:, :, 3] > 0
        if alpha.any():
            lab, n = _ndi.label(alpha)
            sizes = _ndi.sum(alpha, lab, range(1, n + 1))
            big = int(np.argmax(sizes)) + 1
            cy, cx = _ndi.center_of_mass(lab == big)
            if lab[int(cy), int(cx)] == big:
                probe = (int(cx), int(cy))
            else:                       # centroid can fall outside a curved region
                ys, xs = np.nonzero(lab == big)
                probe = (int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    except Exception:
        pass
    _, fr = req(B, f"/api/image/{iid}/flip_region", "POST",
                dict(x=probe[0], y=probe[1], mode="remove"), timeout=180)
    if fr.get("ok"):
        check("remove-region deletes a whole component", fr.get("region_px", 0) > 1000,
              f"{fr.get('region_px'):,} px, source={fr.get('source')}")
        _, u2 = req(B, f"/api/image/{iid}/undo", "POST")
        check("undo restores a removed region", u2.get("ok") is True)
    else:
        skip("remove-region", fr.get("error", "no region under the probe point"))

    # ---- one click must also be able to label something the model did NOT mark.
    # This is the capability the SEM pipeline documents as the one its dataset was
    # missing: every label collected was positive, so there was no way to record
    # "I looked at this and it is not a crack". A region the model half-fires on is a
    # hard negative sitting on the decision boundary.
    try:
        import numpy as _np
        prob = _np.load(os.path.join(PROJECT, "app_data", "images", iid, "prob.npy"),
                        mmap_mode="r")
        weak = _np.argwhere((_np.asarray(prob) > 0.15) & (_np.asarray(prob) < 0.40))
        if not len(weak):
            skip("one click labels a region the model only half-marked",
                 "no weak-activation pixels on the test image")
        else:
            wy, wx = weak[len(weak) // 2]
            _, wr = req(B, f"/api/image/{iid}/flip_region", "POST",
                        dict(x=int(wx), y=int(wy), mode="remove"), timeout=300)
            check("one click labels a region the model only half-marked",
                  bool(wr.get("ok")) and wr.get("source") == "weak activation"
                  and wr.get("not_px", 0) > 0,
                  f"{wr.get('region_px', 0):,} px via {wr.get('source')!r}"
                  if wr.get("ok") else str(wr.get("error"))[:60])
            if wr.get("ok"):
                req(B, f"/api/image/{iid}/undo", "POST")
    except Exception as e:
        check("one click labels a region the model only half-marked", False, str(e))

    # ---- exports
    for ep, magic, label in [("mask.png", b"\x89PNG", "export B&W mask"),
                             ("overlay.png", b"\x89PNG", "export overlay"),
                             ("stats.csv", None, "export CSV")]:
        try:
            st, raw = req(B, f"/api/export/{iid}/{ep}", timeout=300)
            ok = st == 200 and len(raw) > 200 and (magic is None or raw[:4] == magic)
            extra = f"{len(raw)} bytes"
            if ep == "stats.csv":
                head = raw.decode("utf8", "replace")
                ok = ok and "SourceImage,CrackID,Area_px" in head and "SkeletonLength_px" in head
                extra += "; SEM column set" if ok else "; MISSING SEM columns"
            check(label, ok, extra)
        except Exception as e:
            check(label, False, str(e))
    try:
        st, raw = req(B, "/api/export/all.zip", timeout=900)
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(raw))
        names = z.namelist()
        check("export all.zip", st == 200 and "summary.csv" in names,
              f"{len(names)} entries, {len(raw)/1e6:.1f} MB")
    except Exception as e:
        check("export all.zip", False, str(e))

    # ---- B&W polarity: crack must be BLACK
    try:
        from PIL import Image
        _, raw = req(B, f"/api/export/{iid}/mask.png", timeout=300)
        a = np.array(Image.open(io.BytesIO(raw)))
        vals = sorted(np.unique(a).tolist())
        check("B&W mask is binary with crack=black", vals == [0, 255] or vals in ([0], [255]),
              f"values {vals}, black {float((a==0).mean())*100:.1f}%")
    except Exception as e:
        check("B&W mask is binary with crack=black", False, str(e))

    # ---- threshold + postprocess actually change the output
    try:
        _, m1 = req(B, f"/api/image/{iid}/mask.png?threshold=0.20", timeout=120)
        _, m2 = req(B, f"/api/image/{iid}/mask.png?threshold=0.80", timeout=120)
        check("threshold slider changes the mask", m1 != m2,
              f"{len(m1)} vs {len(m2)} bytes")
        _, p1 = req(B, f"/api/image/{iid}/mask.png?postprocess=0", timeout=120)
        _, p2 = req(B, f"/api/image/{iid}/mask.png?postprocess=1", timeout=300)
        check("post-process toggle changes the mask", p1 != p2)
    except Exception as e:
        check("threshold / post-process toggles", False, str(e))

    # ---- the status bar must describe the mask actually on screen.
    # api_stats used to call effective_mask() with no arguments, so it always
    # reported threshold 0.50 while the user looked at something else.
    try:
        _, s20 = req(B, f"/api/image/{iid}/stats?threshold=0.20", timeout=180)
        _, s80 = req(B, f"/api/image/{iid}/stats?threshold=0.80", timeout=180)
        a20, a80 = s20.get("area_fraction"), s80.get("area_fraction")
        check("stats honour the sensitivity setting", a20 is not None and a80 is not None and a20 > a80,
              f"{a20:.4f} at 0.20 vs {a80:.4f} at 0.80")
    except Exception as e:
        check("stats honour the sensitivity setting", False, str(e))

    # ---- concurrent strokes must not lose each other.
    # Flask runs threaded and every stroke is a load-modify-save of the whole
    # correction array, so without a per-image lock the later write silently
    # discarded the earlier stroke -- which had already reported "saved".
    try:
        import threading as _th
        req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
        boxes = [[[300, 300]], [[400, 400]], [[500, 500]], [[600, 600]],
                 [[700, 700]], [[800, 800]], [[900, 900]], [[1000, 1000]]]
        errs = []

        def paint(pts):
            try:
                req(B, f"/api/image/{iid}/correction", "POST",
                    dict(mode="crack", radius=15, points=pts), timeout=300)
            except Exception as ex:                      # noqa: BLE001
                errs.append(str(ex))

        ts = [_th.Thread(target=paint, args=(p,)) for p in boxes]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # Measure directly rather than probing with undo, which would itself
        # remove one of the dabs being counted.
        _, lst = req(B, "/api/images")
        me = [m for m in lst.get("images", []) if m["id"] == iid]
        painted = me[0].get("corrected_crack_px", 0) if me else 0
        # The 8 dabs are disjoint disks of radius 15, so with the lock held every
        # one survives: 8 * ~709 px. A lost update shows up as a whole dab missing.
        n_equiv = painted / 709.0
        check("concurrent strokes do not overwrite each other",
              not errs and n_equiv >= 7.5,
              f"{painted:,} px = {n_equiv:.1f} of 8 dabs kept"
              + (f"; errors: {errs[:1]}" if errs else ""))
    except Exception as e:
        check("concurrent strokes do not overwrite each other", False, str(e))

    # ---- the overlay is the most-requested endpoint in the app, so it is cached.
    # Assert identical bytes on repeat and that painting invalidates it, rather than
    # asserting a timing, which would be flaky on a loaded machine.
    try:
        _, o1 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        _, o2 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        # Match on the directory rather than one filename: the cache key encodes
        # threshold, post-processing and now the labels toggle, so hard-coding
        # "0.50_0.png" made this test fail the moment a legitimate key gained a field.
        ovdir = os.path.join(PROJECT, "app_data", "images", iid, "overlays")
        cached = sorted(f for f in os.listdir(ovdir)
                        if f.endswith(".png")) if os.path.isdir(ovdir) else []
        check("overlay is cached and stable on repeat", o1 == o2 and bool(cached),
              f"{len(o1)} bytes, cache {cached if cached else 'MISSING'}")
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="crack", radius=20, points=[[250, 250]]))
        _, o3 = req(B, f"/api/image/{iid}/mask.png?threshold=0.50&postprocess=0", timeout=180)
        check("painting invalidates the cached overlay", o3 != o2,
              f"{len(o2)} -> {len(o3)} bytes")
        req(B, f"/api/image/{iid}/undo", "POST")
    except Exception as e:
        check("overlay caching", False, str(e))

    # ---- model picker moved to check_model_picker(), called BEFORE any upload.
    #      See that function for why placement matters.

    # ---- display.png must be revalidatable. It is the largest payload in the app and had
    #      no ETag and no Last-Modified, so every image switch refetched it in full: 30 MB
    #      for the 32 MP mosaic, 0.64 GB to browse 71 images once.
    try:
        st, _ = req(B, f"/api/image/{iid}/display.png", timeout=300)
        import urllib.request as _u
        r = _u.Request(B.rstrip("/") + f"/api/image/{iid}/display.png")
        with _u.urlopen(r, timeout=300) as resp:
            tag = resp.headers.get("ETag")
            resp.read()
        check("display.png carries an ETag", bool(tag), str(tag))
        if tag:
            r2 = _u.Request(B.rstrip("/") + f"/api/image/{iid}/display.png")
            r2.add_header("If-None-Match", tag)
            try:
                with _u.urlopen(r2, timeout=300) as resp2:
                    code, n = resp2.status, len(resp2.read())
            except urllib.error.HTTPError as e:
                code, n = e.code, len(e.read())
            check("an unchanged display.png revalidates to 304 with no body",
                  code == 304 and n == 0, f"http {code}, {n} bytes")
    except Exception as e:                                      # noqa: BLE001
        check("display.png caching", False, str(e))

    # ---- a crafted image id must never reach the filesystem. Before store.path()
    #      validated its argument, DELETE /api/image/%2e%2e handed shutil.rmtree the whole
    #      app_data directory -- every correction mask, the model registry and the retrain
    #      history in one request.
    try:
        # This check runs before the pruning block that also imports from core, so it must
        # put core on the path itself rather than depend on ordering.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
        import store as _St
        bad = ["..", "../..", ".", "a/../..", "", "/etc/passwd"]
        refused = 0
        for b in bad:
            try:
                _St.path(b)
            except ValueError:
                refused += 1
        check("a crafted image id cannot escape app_data", refused == len(bad),
              f"{refused}/{len(bad)} refused by store.path()")
        real = [m["id"] for m in _St.list_images()]
        check("every real image id still passes validation",
              all(_St.valid_id(i) for i in real), f"{len(real)} ids")
        # And over HTTP it must read as "no such image", not a 500 traceback. req() returns
        # (status, body) for 4xx as well -- it deliberately does not raise on those -- so
        # read the status rather than expecting an exception.
        st, body = req(B, "/api/image/%2e%2e", "DELETE", timeout=30)
        check("DELETE with a traversal id answers 404, not 500", st == 404,
              f"http {st}, {str(body)[:60]}")
    except Exception as e:                                      # noqa: BLE001
        check("crafted image ids are refused", False, str(e))

    # ---- speck pruning: it must actually prune, and it must NEVER be able to remove
    #      crack the user painted by hand. The second is the safety property, and it holds
    #      only because prune_specks runs BEFORE corrections are layered on -- swap those
    #      two lines in effective_mask and this check is what catches it.
    try:
        import numpy as _np
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
        import store as _S
        import pipeline as _P
        check("speck pruning is on by default with a measured threshold",
              getattr(_P, "MIN_BLOB_PX", 0) >= 1000,
              f"MIN_BLOB_PX = {getattr(_P, 'MIN_BLOB_PX', None)}")
        painted = [m for m in _S.list_images() if (_S.correction_counts(m["id"])[0] or 0) > 10000]
        if not painted:
            skip("pruning never deletes hand-painted crack", "no image has painted crack")
        else:
            worst, worst_name = 1.0, ""
            for m in painted[:12]:
                corr = _S.load_npy(m["id"], "correction.npy", mmap=True)
                eff = _P.effective_mask(m["id"])
                if corr is None or eff is None:
                    continue
                pnt = _np.asarray(corr) == 1
                tot = int(pnt.sum())
                if tot:
                    frac = float((pnt & eff).sum()) / tot
                    if frac < worst:
                        worst, worst_name = frac, (m.get("filename") or "")[:34]
                del corr, eff, pnt
            check("pruning never deletes hand-painted crack", worst >= 0.999999,
                  f"worst of {min(len(painted), 12)} images: {worst*100:.4f}% kept "
                  f"({worst_name})")
    except Exception as e:                                      # noqa: BLE001
        check("speck pruning", False, str(e))

    # ---- the retrain scorecard must be persisted, not just live in memory.
    try:
        _, rr = req(B, "/api/retrain_report", timeout=60)
        if not (rr.get("history") or []):
            skip("retrain report is kept after the job is gone",
                 "no retrain has run on this install yet")
        else:
            L = rr.get("last") or {}
            # NOT "incumbent": the gate no longer scores a previous model against a
            # labelled test set, so that field is deliberately absent rather than missing.
            need = ["candidate", "candidate_clean_fp", "incumbent_clean_fp",
                    "deployed", "when"]
            missing = [k for k in need if L.get(k) is None]
            # THE LEAKAGE GUARD, restated.
            #
            # This used to assert held-out < in-sample, which stopped meaning anything once
            # the four reference frames left the training set: that figure is held out too
            # now, so there is no reason either number must sit above the other, and the
            # check began failing on a correct pipeline (0.7589 vs 0.7422).
            #
            # What still detects a leaky split is the FOLD SPREAD. Measured on this data with
            # the deployed architecture: grouping by image gives sd ~0.05, while shuffling
            # pixels into folds gives sd 0.003 -- four genuinely different specimens cannot
            # agree to a thousandth, so a suspiciously tight spread is the fingerprint of
            # train and test sharing neighbourhoods. That is the invariant worth asserting,
            # and unlike the old one it fails on the actual mistake rather than on a
            # relationship between two unrelated numbers.
            ho = L.get("heldout")
            if ho:
                sd = ho.get("std_iou")
                check("held-out IoU is grouped by image, with a fold spread that rules out "
                      "a leaky split",
                      ho.get("grouped_by") == "image" and ho.get("k", 0) >= 2
                      and sd is not None and sd > 0.01,
                      f"{ho.get('k')}-fold by {ho.get('grouped_by')}: {ho.get('mean_iou')} "
                      f"(sd {sd}, worst {ho.get('min_iou')}) -- random pixel folds would "
                      f"read sd ~0.003")
            else:
                skip("held-out IoU is grouped by image, with a fold spread that rules out "
                     "a leaky split",
                     "no cross-validated score recorded for the last retrain yet")
            check("retrain report is kept after the job is gone", not missing,
                  f"{rr.get('n_total')} recorded; last {L.get('stamp')} "
                  f"IoU {(L.get('incumbent') or {}).get('iou', 0):.3f}->"
                  f"{(L.get('candidate') or {}).get('iou', 0):.3f}, "
                  f"bg {(L.get('incumbent_clean_fp') or 0)*100:.2f}%->"
                  f"{(L.get('candidate_clean_fp') or 0)*100:.2f}%"
                  + (f"; MISSING {missing}" if missing else ""))
    except Exception as e:                                      # noqa: BLE001
        check("retrain report is kept after the job is gone", False, str(e))

    # ---- A NAME USED IN A FUNCTION THAT IS NEVER BOUND ANYWHERE IN IT.
    #
    # retrain() runs for over an hour before it reaches the gate, so a NameError in that last
    # block costs the whole run -- which is what happened: an `inc_recipe` reference survived
    # a refactor that deleted its assignment, and the crash came after 68 minutes of
    # gathering, fitting and cross-validating. Python cannot warn about it; a parse can.
    #
    # The FIRST version of this check was itself wrong: it collected only `Name` stores, so it
    # did not know that `def _clf():` binds _clf or that a parameter binds its name, and it
    # reported six false positives. A check that cries wolf gets ignored, so it now collects
    # every binding form in the subtree -- nested defs, every argument, imports, except-as,
    # walrus and comprehension targets are all Name stores already.
    try:
        import ast as _ast, builtins as _bi
        _src = open(os.path.join(PROJECT, "app", "core", "pipeline.py")).read()
        _tree = _ast.parse(_src)
        _mod = set()
        for _n in _tree.body:
            if isinstance(_n, _ast.Assign):
                _mod |= {t.id for t in _n.targets if isinstance(t, _ast.Name)}
            elif isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                _mod.add(_n.name)
            elif isinstance(_n, (_ast.Import, _ast.ImportFrom)):
                _mod |= {(a.asname or a.name).split(".")[0] for a in _n.names}
        _bad = {}
        for _fn in [f for f in _tree.body
                    if isinstance(f, (_ast.FunctionDef, _ast.AsyncFunctionDef))]:
            _bound, _used = set(), {}
            for _node in _ast.walk(_fn):
                if isinstance(_node, _ast.arg):
                    _bound.add(_node.arg)                       # every parameter, any depth
                elif isinstance(_node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                        _ast.ClassDef)):
                    _bound.add(_node.name)                      # nested def/class binds a name
                elif isinstance(_node, (_ast.Import, _ast.ImportFrom)):
                    _bound |= {(a.asname or a.name).split(".")[0] for a in _node.names}
                elif isinstance(_node, _ast.ExceptHandler) and _node.name:
                    _bound.add(_node.name)
                elif isinstance(_node, (_ast.Global, _ast.Nonlocal)):
                    _bound |= set(_node.names)
                elif isinstance(_node, _ast.Name):
                    (_bound.add(_node.id) if isinstance(_node.ctx, _ast.Store)
                     else _used.setdefault(_node.id, _node.lineno))
            for _k, _ln in _used.items():
                if _k not in _bound and _k not in _mod and not hasattr(_bi, _k):
                    _bad[f"{_fn.name}:{_ln}"] = _k
        check("no function in pipeline.py uses a name it never binds", not _bad,
              f"{_bad}" if _bad else
              f"checked {sum(1 for f in _tree.body if isinstance(f, (_ast.FunctionDef, _ast.AsyncFunctionDef)))} functions")
    except Exception as e:                                      # noqa: BLE001
        check("no function in pipeline.py uses a name it never binds", False, str(e))

    # ---- WILL A CLONE ACTUALLY HAVE THE MODEL IT DEFAULTS TO?
    #
    # A retrain writes its model into app_data/, which is gitignored, so shipping one means
    # copying it into models/ and pointing the defaults at it. Forgetting either half leaves
    # a repo that works perfectly on the machine that trained the model and raises
    # FileNotFoundError for everyone else. That is not hypothetical -- it was the state of
    # this repo until the model below was committed.
    try:
        import subprocess as _sp
        import model as _M3
        for _lbl, _pth in (("path_17", _M3.DEFAULT_17), ("path_hybrid", _M3.DEFAULT_HYBRID)):
            _rel = os.path.relpath(_pth, PROJECT)
            _tracked = _sp.run(["git", "ls-files", "--error-unmatch", _rel],
                               cwd=PROJECT, capture_output=True).returncode == 0
            check(f"a clone gets the default {_lbl} model",
                  os.path.exists(_pth) and _tracked,
                  f"{_rel}: on disk={os.path.exists(_pth)}, committed={_tracked}")
    except Exception as e:                                      # noqa: BLE001
        check("a clone gets the default models", False, str(e))

    # store.py spells its recipe tag out as a literal because pipeline imports store, so the
    # two can drift apart silently. If they do, the shipped entry reads as a foreign recipe
    # and every future retrain is gated against an absolute floor instead of its predecessor.
    try:
        import store as _S3
        import pipeline as _P3
        _tag = (_S3._default_registry().get("current") or {}).get("recipe")
        check("the shipped registry entry carries the current recipe tag",
              _tag == _P3.RECIPE,
              f"store.py={_tag!r} pipeline.RECIPE={_P3.RECIPE!r}")
    except Exception as e:                                      # noqa: BLE001
        check("the shipped registry entry carries the current recipe tag", False, str(e))

    # The embedding lookup, checked as arithmetic rather than by eye.
    #
    # Tile seams were a real shipped artifact: with tiles abutting, the worst row on one
    # frame carried 32.8x the median row-to-row probability change, all of it at the y=1023
    # tile boundary. The fix has three separable ways to regress silently, so each is a
    # separate assertion. If TILE_STRIDE ever goes back to TILE the tiles stop overlapping
    # and the seams return with no other symptom; if the window stops vanishing at a tile's
    # own edge the field jumps wherever a tile joins the blend (a 1e-3 weight floor alone
    # put a 0.4999 step back in); and if the weights stop normalising, every embedding is
    # silently scaled.
    try:
        import numpy as _np
        import model as _M4
        _big = (3 * _M4.TILE, 3 * _M4.TILE)
        _xs = sorted({t[2] for t in _M4.tiles(_big, stride=_M4.TILE_STRIDE)})
        _laps = [_M4.TILE - (_xs[i + 1] - _xs[i]) for i in range(len(_xs) - 1)]
        check("embedding tiles overlap, so the seams can be blended out",
              _M4.TILE_STRIDE < _M4.TILE and _laps and min(_laps) > 0,
              f"stride {_M4.TILE_STRIDE} of {_M4.TILE}, overlaps {_laps}")

        # all-ones in must give all-ones out, or the weights are not a partition of unity
        _c2 = _np.array([[0, 0], [0, _M4.TILE_STRIDE]], _np.int32)
        _ones = _np.ones((2, 2, 64, 64), _np.float16)
        _rr = _np.array([0, 0, 500, _M4.TILE - 1])
        _cc = _np.array([0, 900, 1000, _M4.TILE + 500])
        _err = float(_np.abs(_M4.emb_rows(_c2, _ones, _rr, _cc) - 1.0).max())
        check("blend weights normalise to one", _err < 1e-5, f"max error {_err:.2e}")

        # a hard step between two tiles must come out as a ramp, not a step. Checked on the
        # frame's top row, where every window is near zero and a weight floor would show up.
        _step = _np.concatenate([_np.zeros((1, 2, 64, 64), _np.float16),
                                 _np.ones((1, 2, 64, 64), _np.float16)])
        _x = _np.arange(0, _M4.TILE_STRIDE + _M4.TILE)
        _worst = 0.0
        for _row in (0, _M4.TILE // 2, _M4.TILE - 1):
            _v = _M4.emb_rows(_c2, _step, _np.full_like(_x, _row), _x)[:, 0]
            _worst = max(_worst, float(_np.abs(_np.diff(_v)).max()))
        check("a step between tiles comes out as a ramp, not a seam",
              _worst < 0.05, f"largest single-pixel jump {_worst:.4f} (last-wins would be 1.0)")

        # one tile covering the pixel must reduce to a plain lookup, or interiors are wrong
        _rng = _np.random.default_rng(0)
        _e1 = _rng.standard_normal((1, 4, 64, 64)).astype(_np.float16)
        _c1 = _np.array([[0, 0]], _np.int32)
        _r1 = _np.array([0, 5, 500, _M4.TILE - 1]); _k1 = _np.array([0, 7, 600, _M4.TILE - 1])
        _d = float(_np.abs(_M4.emb_rows(_c1, _e1, _r1, _k1)
                           - _M4.interp_tile(_e1[0], _r1, _k1)).max())
        check("a single covering tile reduces to a plain lookup", _d < 1e-5,
              f"max difference {_d:.2e}")
    except Exception as e:                                      # noqa: BLE001
        check("the embedding lookup blends without seams", False, str(e))

    # "Tight crack boundary" must narrow the mask, not redraw it. The exported mask is as
    # wide as the brush that labelled it -- measured, the strokes on these frames are 2.6x to
    # 15x wider than the dark crack core they mark -- and no probability threshold recovers
    # the width, because it was never in the labels. Otsu inside the accepted region uses the
    # image instead. Two properties have to hold: it can only ever REMOVE area (a subset of
    # what was already accepted), and it must keep the dark core rather than trimming
    # arbitrarily.
    try:
        import numpy as _np9
        import pipeline as _P9
        import store as _S9
        _checked = _subset = _kept = 0
        for _m9 in _S9.list_images():
            if "SELFTEST" in (_m9.get("filename") or ""):
                continue
            _wide = _P9.effective_mask(_m9["id"], corrections="gate")
            if _wide is None or not _wide.any():
                continue
            _img9 = _S9.load_npy(_m9["id"], "img.npy")
            if _img9 is None:
                continue
            _tight9 = _P9.effective_mask(_m9["id"], corrections="gate", tight=True)
            _checked += 1
            if bool((_tight9 & ~_wide).any()):
                break                       # added area: not a narrowing
            _subset += 1
            # The contract is not "keeps 90%". Narrowing rests on crack being locally
            # darker, which is false on some frames. It is: either the dark core survives at
            # TIGHTEN_MIN_CORE, or the frame was detected as one where the rule fails and
            # left alone entirely. Silently deleting the crack is what must not happen.
            _im9 = _np9.asarray(_img9, _np9.float32)
            _core = _wide & (_im9 <= _np9.percentile(_im9[_wide], 20))
            _ratio = ((_tight9 & _core).sum() / _core.sum()) if _core.any() else 1.0
            if _ratio >= _P9.TIGHTEN_MIN_CORE or bool((_tight9 == _wide).all()):
                _kept += 1
            if _checked >= 4:
                break
        check("tight boundary narrows, or declines when it would delete the crack",
              _checked > 0 and _subset == _checked and _kept == _checked,
              f"{_checked} frames: {_subset} narrowed only, {_kept} kept the core or declined")
    except Exception as e:                                      # noqa: BLE001
        check("tight boundary narrows, or declines when it would delete the crack", False, str(e))

    # The operating point lives in ONE place. It used to be five literals plus two `> 0.5`
    # comparisons inside the gate's own false-positive axis, and that last pair is why this
    # check exists: had the served threshold moved without them, the gate would have kept
    # scoring candidates at 0.50 while the app served something else -- measuring a model
    # nobody was running, and reporting it as the shipped number.
    try:
        import re as _re11
        import pipeline as _P11
        _bad = []
        for _rel11 in ("app/core/pipeline.py", "app/server.py"):
            _src11 = open(os.path.join(PROJECT, _rel11)).read()
            for _pat11 in (r'threshold\s*=\s*0\.\d', r'threshold"\s*,\s*0\.\d',
                           r'>\s*0\.5\b'):
                for _m11 in _re11.finditer(_pat11, _src11):
                    _bol11 = _src11.rfind("\n", 0, _m11.start()) + 1
                    _hash11 = _src11.find("#", _bol11, _m11.start())
                    if _hash11 != -1:
                        continue          # inside a comment -- prose about the fix, not code
                    _ln11 = _src11[:_m11.start()].count("\n") + 1
                    _bad.append(f"{os.path.basename(_rel11)}:{_ln11} {_m11.group(0)!r}")
        check("the decision threshold is defined in one place",
              not _bad and 0.05 < _P11.DEFAULT_THRESHOLD < 0.95,
              (f"DEFAULT_THRESHOLD={_P11.DEFAULT_THRESHOLD}" if not _bad
               else "hardcoded: " + "; ".join(_bad[:4])))
    except Exception as e:                                      # noqa: BLE001
        check("the decision threshold is defined in one place", False, str(e))

    # No module-level constant assigned twice. CV_TRAIN_CAP was set to 400000 and then to
    # 90000 three lines later, so the intended cap was dead on arrival and the effective one
    # was whichever line came last -- silently, for two days, while the comment above them
    # described the value that never ran. Nothing catches that: it is valid Python, it does
    # not shadow a name, and the tests still pass because a wrong-but-consistent constant
    # produces wrong-but-consistent numbers. Only a duplicate-assignment check sees it.
    #
    # Names re-bound inside functions or under `if` are normal and not counted -- only
    # straight-line module scope, where a second assignment can never be intentional.
    try:
        import ast as _ast10
        _dupes = []
        for _rel10 in ("app/core/pipeline.py", "app/core/model.py", "app/core/store.py",
                       "app/server.py"):
            _tree10 = _ast10.parse(open(os.path.join(PROJECT, _rel10)).read())
            _seen10 = {}
            for _n10 in _tree10.body:                    # module scope only, not nested
                if not isinstance(_n10, _ast10.Assign):
                    continue
                for _t10 in _n10.targets:
                    if not isinstance(_t10, _ast10.Name):
                        continue
                    if not _t10.id.isupper():            # constants; lowercase gets rebound
                        continue
                    if _t10.id in _seen10:
                        _dupes.append(f"{os.path.basename(_rel10)}:{_t10.id} "
                                      f"(lines {_seen10[_t10.id]} and {_t10.lineno})")
                    _seen10[_t10.id] = _t10.lineno
        check("no module-level constant is assigned twice", not _dupes,
              "; ".join(_dupes) if _dupes else "checked 4 modules")
    except Exception as e:                                      # noqa: BLE001
        check("no module-level constant is assigned twice", False, str(e))

    # A brush stroke must be continuous even when the pointer outruns it. The painter used
    # to stamp one disc per reported point and nothing in between, so a stroke was only
    # continuous if the browser happened to deliver positions closer together than the brush
    # diameter -- and on a 23 MP frame it does not. The result was a dotted line of separate
    # discs in the exported mask, reported as "black dots that don't look like crack", plus
    # the beaded stroke edges that looked like brush geometry and were really gaps.
    try:
        import numpy as _np8
        from skimage.measure import label as _lab8
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import server as _SV8
        _H8 = _W8 = 600
        _r8 = 20
        _a8 = _np8.zeros((_H8, _W8), _np8.uint8)
        _SV8._sweep(_a8, 1, _r8, [[100, 300], [300, 300], [500, 300]], _H8, _W8)
        _n8 = int(_lab8(_a8 == 1, connectivity=2).max())
        # and a lone point must still be exactly a disc, or single clicks change shape
        _c8 = _np8.zeros((_H8, _W8), _np8.uint8)
        _SV8._sweep(_c8, 1, _r8, [[300, 300]], _H8, _W8)
        _yy8, _xx8 = _np8.ogrid[-_r8:_r8 + 1, -_r8:_r8 + 1]
        _d8 = _np8.zeros((_H8, _W8), _np8.uint8)
        _d8[300 - _r8:300 + _r8 + 1, 300 - _r8:300 + _r8 + 1][
            (_xx8 * _xx8 + _yy8 * _yy8) <= _r8 * _r8] = 1
        check("a fast stroke paints one continuous mark, not a row of discs",
              _n8 == 1 and _np8.array_equal(_c8, _d8),
              f"{_n8} component(s) from 3 points 200 px apart at radius {_r8}; "
              f"single click still a disc: {_np8.array_equal(_c8, _d8)}")
    except Exception as e:                                      # noqa: BLE001
        check("a fast stroke paints one continuous mark, not a row of discs", False, str(e))

    # No speck survives into an export. prune_specks promises no crack blob under
    # MIN_BLOB_PX, and corrections used to be applied AFTER that prune, which quietly undid
    # it: an eraser stroke does not only remove area, it cuts THROUGH blobs and leaves the
    # offcuts behind as separate sub-floor components. Measured on b2_343_75_LARGE before the
    # fix: 19 components with corrections off, 70 with them on, 51 of those under 200 px --
    # visible in an exported mask as tiny black dots that look nothing like crack.
    try:
        import numpy as _np7
        import pipeline as _P7
        import store as _S7
        from skimage.measure import label as _lab7
        _worst, _n7, _checked = None, 0, 0
        for _m7 in _S7.list_images():
            if "SELFTEST" in (_m7.get("filename") or ""):
                continue
            # tight=True: what the export actually serves by default. Testing the
            # untightened path would pass while the real deliverable regressed.
            _mask7 = _P7.effective_mask(_m7["id"], corrections="gate", tight=True)
            if _mask7 is None:
                continue
            _checked += 1
            _s7 = _np7.bincount(_lab7(_mask7, connectivity=2).ravel())[1:]
            if len(_s7) == 0:
                continue
            # Below the full floor, an elongated thread is crack and a roundish blob is a
            # dot. Tightening splits wide bands into threads, so size alone cannot judge it.
            from skimage.measure import regionprops as _rp7
            _k7 = sum(1 for _r7 in _rp7(_lab7(_mask7, connectivity=2))
                      if _r7.area < _P7.MIN_BLOB_PX
                      and _r7.major_axis_length / max(_r7.minor_axis_length, 1e-6) < 3.0)
            if _k7 > _n7:
                _n7, _worst = _k7, (_m7.get("filename") or "")[22:50]
            if _checked >= 8:      # a sample: the full sweep is minutes, this is seconds
                break
        check("an exported mask carries no roundish sub-floor blob",
              _n7 == 0,
              f"{_n7} roundish sub-{_P7.MIN_BLOB_PX}px component(s) over {_checked} frames"
              + (f", worst {_worst}" if _worst else ""))
    except Exception as e:                                      # noqa: BLE001
        check("an exported mask carries no blob under the speck floor", False, str(e))

    # Staging files left by writes that were KILLED rather than raised. The atomic-write
    # helpers unlink their temp on exception, but no handler runs for SIGKILL or a power cut,
    # and nothing reads or reaps those files -- so they accumulate invisibly. This install
    # reached 271 of them, 12.8 GB, after one day of interrupted retrains. The sweep must
    # take the stale ones and leave anything recent, because a file being written right now
    # looks exactly the same apart from its age.
    try:
        import store as _S6
        import numpy as _np6
        import time as _t6
        _d6 = os.path.join(_S6.IMAGES, "SELFTEST_SWEEP__probe")
        os.makedirs(_d6, exist_ok=True)
        _old = os.path.join(_d6, "prob.npy.999999.888888.tmp")
        _new = os.path.join(_d6, "prob.npy.999999.888889.tmp")
        try:
            for _p6 in (_old, _new):
                with open(_p6, "wb") as _fh6:
                    _np6.save(_fh6, _np6.zeros((4, 4), _np6.float32))
            os.utime(_old, (_t6.time() - 7200, _t6.time() - 7200))
            _n6, _b6 = _S6.sweep_stale_temps()
            _gone = not os.path.exists(_old)
            _kept = os.path.exists(_new)
            check("unfinished staging files are reaped, recent ones spared",
                  _gone and _kept, f"stale removed={_gone}, in-flight kept={_kept}")
        finally:
            for _p6 in (_old, _new):
                try:
                    os.unlink(_p6)
                except OSError:
                    pass
            try:
                os.rmdir(_d6)
            except OSError:
                pass
    except Exception as e:                                      # noqa: BLE001
        check("unfinished staging files are reaped, recent ones spared", False, str(e))

    # An embedding cache built before the tiles overlapped cannot be blended, and serving
    # from one would reintroduce the seams with nothing to show for it. read_emb has to
    # reject an untagged cache rather than trust it.
    try:
        import numpy as _np5
        import model as _M5
        import tempfile as _tf5
        with _tf5.TemporaryDirectory() as _d5:
            _p5 = os.path.join(_d5, "emb.npz")
            _np5.savez(_p5, coords=_np5.zeros((1, 2), _np5.int32),
                       emb=_np5.zeros((1, 2, 64, 64), _np5.float16))
            _untagged = _M5.read_emb(_p5) is None and not _M5.emb_is_current(_p5)
            _M5.write_emb(_p5, _np5.zeros((1, 2), _np5.int32),
                          _np5.zeros((1, 2, 64, 64), _np5.float16))
            _tagged = _M5.read_emb(_p5) is not None and _M5.emb_is_current(_p5)
        check("a pre-overlap embedding cache is rejected, a current one accepted",
              _untagged and _tagged,
              f"untagged rejected={_untagged}, tagged accepted={_tagged}")
    except Exception as e:                                      # noqa: BLE001
        check("a pre-overlap embedding cache is rejected, a current one accepted",
              False, str(e))

    # ---- the UNCACHED half of a model switch, checked structurally.
    #
    # The check above can only ever exercise the cached path: it deliberately picks a
    # model already computed for every image, because the alternative is a real
    # 70-image prediction pass. That left the branch that actually predicts untested,
    # and it broke -- a comment reflow dedented P.ingest out of the `for iid in todo`
    # loop, so a switch walked the progress bar over every pending image, predicted only
    # the LAST one, and returned predicted=todo claiming all of them. Every other image
    # silently kept the previous model's mask while the UI reported success.
    #
    # Cheap to assert structurally, and this is the assertion that would have caught it.
    try:
        import ast
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        fn = next(f for f in ast.walk(ast.parse(open(src).read()))
                  if isinstance(f, ast.FunctionDef) and f.name == "api_model_select")
        work = next(n for n in ast.walk(fn)
                    if isinstance(n, ast.FunctionDef) and n.name == "work")
        loop = next(st for st in work.body if isinstance(st, ast.For))
        body = [ast.unparse(b) for b in loop.body]
        check("a model switch predicts every pending image, not just the last",
              any("P.ingest" in b for b in body),
              f"loop body: {[b[:38] for b in body]}")
    except Exception as e:
        check("a model switch predicts every pending image, not just the last",
              False, f"{type(e).__name__}: {e}")

    # ---- reset
    _, c = req(B, f"/api/image/{iid}/correction", "POST", dict(mode="clear"))
    check("reset clears this image's corrections", c.get("crack_px") == 0 and c.get("not_px") == 0)
    _, u3 = req(B, f"/api/image/{iid}/undo", "POST")
    check("undo restores after reset", u3.get("ok") is True,
          f"crack {u3.get('crack_px'):,} px back")

    # ---- retrain (optional, slow)
    if args.retrain:
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="crack", radius=40, points=[[500, 700], [560, 720], [620, 740]]))
        req(B, f"/api/image/{iid}/correction", "POST",
            dict(mode="erase", radius=50, points=[[1200, 300], [1260, 320], [1320, 340]]))
        _, rt = req(B, "/api/retrain", "POST", dict(deploy=False))
        j = wait_job(B, rt["job"], "retrain")
        res = j.get("result") or {}
        check("retrain ran", j["state"] == "done", j.get("error") or f"{j.get('seconds')}s")
        if res.get("ok"):
            info = res.get("info", {})
            check("ground truth reached training (no width-dropped blocks)",
                  info.get("blocks_dropped_for_width") == 0,
                  f"dropped {info.get('blocks_dropped_for_width')}")
            check("class balance sane", 0.3 <= info.get("crack_fraction", 0) <= 0.7,
                  f"{info.get('crack_fraction', 0)*100:.1f}% crack")
            check("candidate was validated against ground truth", "candidate" in res,
                  f"incumbent {res.get('incumbent', {}).get('iou', 0):.3f} -> "
                  f"candidate {res.get('candidate', {}).get('iou', 0):.3f}")
        else:
            check("retrain refused for a stated reason", bool(res.get("error")),
                  str(res.get("error"))[:90])
    else:
        skip("retrain", "pass --retrain to include it (slow)")

    if not args.keep:
        try:
            req(B, f"/api/image/{iid}", "DELETE")
            print(f"\n  cleaned up test image {iid}")
        except Exception:
            pass

    print(f"\n{'='*66}")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
