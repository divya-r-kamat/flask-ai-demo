import os
import uuid
import re
import csv
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, get_flashed_messages
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from asgiref.wsgi import WsgiToAsgi

# --- config ---
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.secret_key = "change-this-in-prod"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# --- helpers ---
def to_uint8(x):
    x = np.nan_to_num(x)
    if x.dtype == np.uint8:
        return x
    x = np.abs(x)
    mx = x.max() if x.size else 0
    if mx != 0:
        x = (x / mx) * 255.0
    return x.astype(np.uint8)

# --- routes ---
@app.route("/")
def index():
    # default empty results; active_tab can be passed via query param or will default to 'filter'
    active_tab = request.args.get("tab", "filter")
    return render_template("index.html", active_tab=active_tab)

# ---------------- Image Filter (convolution + gradients) ----------------
@app.route("/filter", methods=["POST"])
def filter_image():
    file = request.files.get("file")
    kernel_text = request.form.get("kernel", "").strip()
    if not file or file.filename == "":
        flash("No image provided")
        return redirect(url_for("index", tab="filter"))
    if not allowed_file(file.filename):
        flash("Unsupported file type")
        return redirect(url_for("index", tab="filter"))

    # parse kernel (expects 9 numbers)
    try:
        vals = [v for v in re.split(r"[\s,]+", kernel_text) if v != ""]
        kernel = np.array([float(x) for x in vals], dtype=np.float32)
        if kernel.size != 9:
            raise ValueError("Kernel must contain exactly 9 numbers")
        kernel = kernel.reshape((3, 3))
    except Exception as e:
        flash(f"Invalid kernel: {e}")
        return redirect(url_for("index", tab="filter"))

    uid = uuid.uuid4().hex
    fname = uid + "_" + secure_filename(file.filename)
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(in_path)

    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        flash("Failed to read image")
        return redirect(url_for("index", tab="filter"))

    filtered = cv2.filter2D(img, -1, kernel)

    # gradients on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    # convert to uint8 for saving
    gx_u = to_uint8(gx)
    gy_u = to_uint8(gy)
    gmag_u = to_uint8(gmag)

    base = uid
    out_orig = os.path.join(app.config["RESULT_FOLDER"], f"{base}_orig.png")
    out_filtered = os.path.join(app.config["RESULT_FOLDER"], f"{base}_filtered.png")
    out_gx = os.path.join(app.config["RESULT_FOLDER"], f"{base}_grad_x.png")
    out_gy = os.path.join(app.config["RESULT_FOLDER"], f"{base}_grad_y.png")
    out_gmag = os.path.join(app.config["RESULT_FOLDER"], f"{base}_grad_mag.png")

    cv2.imwrite(out_orig, img)
    cv2.imwrite(out_filtered, filtered)
    cv2.imwrite(out_gx, gx_u)
    cv2.imwrite(out_gy, gy_u)
    cv2.imwrite(out_gmag, gmag_u)

    channels = img.shape[2] if img.ndim == 3 else 1
    params = {
        "kernel_shape": "3x3",
        "kernel_params": int(kernel.size * channels),
        "image_shape": f"{img.shape[1]}x{img.shape[0]}x{channels}",
    }

    images = [os.path.basename(p) for p in [out_orig, out_filtered, out_gx, out_gy, out_gmag]]
    filter_results = {"images": images, "params": params}
    return render_template("index.html", active_tab="filter", filter_results=filter_results)

# ---------------- Mean Images ----------------
@app.route("/mean", methods=["POST"])
def mean_images():
    files = request.files.getlist("files")
    files = [f for f in files if f and f.filename != "" and allowed_file(f.filename)]
    if not files:
        flash("No valid image files uploaded")
        return redirect(url_for("index", tab="mean"))

    uid = uuid.uuid4().hex
    read_images = []
    sizes = []
    saved_inputs = []
    for i, f in enumerate(files):
        fname = f"{uid}_{i}_" + secure_filename(f.filename)
        fp = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(fp)
        saved_inputs.append(os.path.basename(fp))
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        read_images.append(img)
        sizes.append((img.shape[1], img.shape[0]))  # w,h

    if not read_images:
        flash("Couldn't read any images after upload")
        return redirect(url_for("index", tab="mean"))

    min_w = min(w for w, h in sizes)
    min_h = min(h for w, h in sizes)
    resized = [cv2.resize(im, (min_w, min_h), interpolation=cv2.INTER_AREA) for im in read_images]
    stack = np.stack(resized, axis=0).astype(np.float32)
    mean_img = np.mean(stack, axis=0).astype(np.uint8)

    mean_path = os.path.join(app.config["RESULT_FOLDER"], f"{uid}_mean.png")
    cv2.imwrite(mean_path, mean_img)

    blended = []
    for idx, im in enumerate(resized):
        b = cv2.addWeighted(im, 0.5, mean_img, 0.5, 0)
        bp = os.path.join(app.config["RESULT_FOLDER"], f"{uid}_blend_{idx}.png")
        cv2.imwrite(bp, b)
        blended.append(os.path.basename(bp))

    images = [os.path.basename(mean_path)] + blended
    params = {
        "num_images": len(resized),
        "mean_shape": f"{min_w}x{min_h}x{resized[0].shape[2]}",
        "pixels_processed": min_w * min_h * resized[0].shape[2] * len(resized),
    }
    mean_results = {"images": images, "params": params, "inputs": saved_inputs}
    return render_template("index.html", active_tab="mean", mean_results=mean_results)

# ---------------- One-hot ----------------
@app.route("/onehot", methods=["POST"])
def onehot():
    text = request.form.get("words", "").strip()
    if not text:
        flash("Please enter words (space/comma separated)")
        return redirect(url_for("index", tab="onehot"))
    tokens = [t for t in re.split(r"[\s,]+", text) if t != ""]
    vocab = list(dict.fromkeys(tokens))
    matrix = []
    for t in tokens:
        row = [1 if t == v else 0 for v in vocab]
        matrix.append(row)

    uid = uuid.uuid4().hex
    csv_path = os.path.join(app.config["RESULT_FOLDER"], f"{uid}_onehot.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["token"] + vocab)
        for tok, row in zip(tokens, matrix):
            writer.writerow([tok] + row)

    params = {"vocab_size": len(vocab), "tokens_count": len(tokens)}
    onehot_results = {"vocab": vocab, "matrix": matrix, "tokens": tokens, "csv": os.path.basename(csv_path), "params": params}
    return render_template("index.html", active_tab="onehot", onehot_results=onehot_results)

# ---------------- Token Count ----------------
@app.route("/tokens", methods=["POST"])
def tokens():
    text = request.form.get("paragraph", "").strip()
    tokens_list = text.split()
    count = len(tokens_list)
    token_results = {"count": count, "tokens": tokens_list}
    return render_template("index.html", active_tab="tokens", token_results=token_results)

# ---------------- Static results ----------------
@app.route("/results/<path:filename>")
def send_result(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

# ASGI adapter so you can run this with uvicorn if you want:
asgi_app = WsgiToAsgi(app)

if __name__ == "__main__":
    # dev friendly
    app.run(debug=True, port=5000)
