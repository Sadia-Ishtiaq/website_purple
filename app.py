from flask import Flask, render_template, request, send_from_directory, redirect, session, flash, url_for, jsonify
import os
import sqlite3
import shutil
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import cv2
from PIL import Image
import time
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors

# ---------------- APP CONFIG ---------------- #
app = Flask(__name__)
app.secret_key = "brain-tumor-secure-secret-key-2024"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ---------------- MODELS ---------------- #
MODEL_PATH = "resnet50_brain_tumor_best (1).h5"
model = None
yolo_model = None

# Load ResNet50 model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"✅ ResNet50 Model loaded successfully")
    else:
        print(f"⚠️ ResNet50 Model file not found. Using mock predictions.")
except Exception as e:
    print(f"❌ Error loading ResNet50 model: {e}")

# Load YOLO model
try:
    yolo_model_path = "yolo_model/best.pt"
    if os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLOv8 Model loaded successfully")
    else:
        print(f"⚠️ YOLO model not found. Localization will use basic detection.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Tumor information dictionary for reports
TUMOR_INFO = {
    "Glioma": "Glioma is a type of tumor that occurs in the brain and spinal cord...",
    "Meningioma": "Meningioma is a tumor that arises from the meninges — the membranes...",
    "Pituitary": "Pituitary tumors are abnormal growths that develop in your pituitary gland...",
    "No Tumor": "No tumor detected. The brain structure appears normal..."
}

# ---------------- LOGIN REQUIRED ---------------- #
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to access this page", "error")
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated

# ---------------- DATABASE ---------------- #
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Scans table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            result_filename TEXT,
            result TEXT NOT NULL,
            confidence REAL,
            processing_time REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# ---------------- HELPER FUNCTIONS ---------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_tumor_with_yolo(image_path):
    if yolo_model is None:
        print("⚠️ YOLO model not available, skipping localization")
        return None
    
    try:
        results = yolo_model.predict(
            source=image_path,
            conf=0.25,
            save=True,
            project='runs/detect',
            name='predict',
            exist_ok=True,
            verbose=False
        )
        time.sleep(1)
        if results and len(results) > 0:
            output_dir = results[0].save_dir
            # Take first jpg/png in dir
            for file in os.listdir(output_dir):
                if file.endswith(('.jpg','.png','.jpeg')):
                    src = os.path.join(output_dir, file)
                    dst_name = f"localized_{int(time.time())}_{file}"
                    dst = os.path.join(RESULT_FOLDER, dst_name)
                    shutil.copy2(src, dst)
                    return dst_name
        return None
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        return None

def save_scan(user_id, filename, result_filename, result, confidence, processing_time):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        """INSERT INTO scans 
           (user_id, filename, result_filename, result, confidence, processing_time) 
           VALUES (?,?,?,?,?,?)""",
        (user_id, filename, result_filename, result, confidence, processing_time)
    )
    conn.commit()
    scan_id = c.lastrowid
    conn.close()
    return scan_id

def get_user_scans(user_id, limit=None):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    if limit:
        c.execute(
            "SELECT * FROM scans WHERE user_id = ? ORDER BY date DESC LIMIT ?",
            (user_id, limit)
        )
    else:
        c.execute(
            "SELECT * FROM scans WHERE user_id = ? ORDER BY date DESC",
            (user_id,)
        )
    scans = c.fetchall()
    conn.close()
    return scans

def get_scan_stats(user_id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ?", (user_id,))
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result != 'No Tumor'", (user_id,))
    detected = c.fetchone()[0]
    stats = {}
    for cls in CLASSES:
        c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = ?", (user_id, cls))
        stats[cls] = c.fetchone()[0]
    conn.close()
    return total, detected, total-detected, stats

# ---------------- AUTH ROUTES ---------------- #
@app.route("/register", methods=["GET","POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method=="POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")
        if not all([name,email,password,confirm]):
            flash("All fields are required","error")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match","error")
            return redirect(url_for("register"))
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)",
                      (name,email,generate_password_hash(password)))
            conn.commit()
            flash("Registration successful! Please login.","success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists","error")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method=="POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?",(email,))
        user=c.fetchone()
        conn.close()
        if user and check_password_hash(user[3],password):
            session["user_id"]=user[0]
            session["user_name"]=user[1]
            session["user_email"]=user[2]
            flash(f"Welcome back, {user[1]}!","success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password","error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out","success")
    return redirect(url_for("login"))

# ---------------- MAIN ROUTES ---------------- #
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET","POST"])
def contact():
    if request.method=="POST":
        flash("Thank you for your message! We'll respond within 24 hours.","success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/features")
def features():
    return render_template("features.html")

# ---------------- DASHBOARD & HISTORY ---------------- #
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    total, detected, healthy, stats = get_scan_stats(user_id)
    recent_scans = get_user_scans(user_id,limit=5)
    return render_template("dashboard.html",user_name=session["user_name"],
                           total_scans=total,detected=detected,healthy=healthy,
                           stats=stats,recent_scans=recent_scans,classes=CLASSES)

@app.route("/history")
@login_required
def history():
    user_id=session["user_id"]
    scans=get_user_scans(user_id)
    return render_template("history.html",scans=scans,classes=CLASSES)

# ---------------- PREDICTION ---------------- #
@app.route("/predict",methods=["POST"])
@login_required
def predict():
    start_time=time.time()
    if 'file' not in request.files:
        flash("No file selected","error")
        return redirect(url_for("dashboard"))
    file=request.files['file']
    if file.filename=='':
        flash("No file selected","error")
        return redirect(url_for("dashboard"))
    if not allowed_file(file.filename):
        flash("File type not allowed","error")
        return redirect(url_for("dashboard"))
    filename=secure_filename(file.filename)
    unique_filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    filepath=os.path.join(UPLOAD_FOLDER,unique_filename)
    file.save(filepath)
    # ResNet prediction
    if model:
        img=load_img(filepath,target_size=(128,128))
        arr=np.expand_dims(img_to_array(img)/255.0,axis=0)
        preds=model.predict(arr,verbose=0)
        idx=np.argmax(preds[0])
        label=CLASSES[idx]
        confidence=float(preds[0][idx])
    else:
        idx=random.randint(0,3)
        label=CLASSES[idx]
        confidence=random.uniform(0.75,0.99)
    # YOLO localization
    result_filename=None
    if label!="No Tumor":
        result_filename=detect_tumor_with_yolo(filepath)
    processing_time=time.time()-start_time
    user_id=session["user_id"]
    scan_id=save_scan(user_id,unique_filename,result_filename,label,confidence,processing_time)
    return render_template("result.html",uploaded_img=unique_filename,
                           result_img=result_filename,label=label,
                           confidence=f"{confidence*100:.2f}%",
                           scan_id=scan_id,processing_time=f"{processing_time:.2f}")

# ---------------- FILE SERVING ---------------- #
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
@login_required
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# ---------------- RUN ---------------- #
if __name__=="__main__":
    if os.path.exists("database.db"):
        os.remove("database.db")
    init_db()
    app.run(debug=True,host='0.0.0.0',port=5000)
