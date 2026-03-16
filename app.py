from flask import Flask, render_template, request, send_from_directory, redirect, session, flash, url_for
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
    "Glioma": "Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
    "Meningioma": "Meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign).",
    "Pituitary": "Pituitary tumors are abnormal growths that develop in your pituitary gland. Most pituitary tumors are noncancerous (benign) growths that remain in the pituitary gland.",
    "No Tumor": "No tumor detected. The brain structure appears normal with no abnormal growths or masses visible in the MRI scan."
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
    """YOLO detection with proper output handling - NO HEATMAP FALLBACK"""
    if yolo_model is None:
        print("⚠️ YOLO model not available, skipping localization")
        return None
    
    try:
        print(f"🔍 Processing image with YOLO: {image_path}")
        
        # Get original filename info
        img_name = os.path.basename(image_path)
        img_name_without_ext = os.path.splitext(img_name)[0]
        img_ext = os.path.splitext(img_name)[1]
        
        # Run YOLO prediction
        results = yolo_model.predict(
            source=image_path,
            conf=0.25,
            save=True,
            project='runs/detect',
            name='predict',
            exist_ok=True,
            verbose=False
        )
        
        # Wait a moment for file to be written
        time.sleep(1.5)
        
        # Get the actual output directory from YOLO
        if results and len(results) > 0:
            output_dir = results[0].save_dir
            print(f"📁 YOLO output directory: {output_dir}")
            
            # Check for the most common YOLO output patterns
            possible_filenames = [
                img_name,
                f"{img_name_without_ext}{img_ext}",
                f"{img_name_without_ext}.jpg",
                f"{img_name_without_ext}.jpeg",
                f"{img_name_without_ext}.png",
                "image0.jpg",
                f"image0{img_ext}",
            ]
            
            # Also check for the image path from results
            if hasattr(results[0], 'path') and results[0].path:
                yolo_saved_path = results[0].path
                if os.path.exists(yolo_saved_path):
                    possible_filenames.append(os.path.basename(yolo_saved_path))
            
            # Search for the file in output directory
            found_file = None
            for filename in possible_filenames:
                potential_path = os.path.join(output_dir, filename)
                if os.path.exists(potential_path):
                    found_file = potential_path
                    print(f"✅ Found YOLO output: {potential_path}")
                    break
            
            # If not found, search recursively
            if not found_file and os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            found_file = os.path.join(root, file)
                            print(f"✅ Found YOLO output via search: {found_file}")
                            break
                    if found_file:
                        break
            
            # If we found the file, copy it to results folder
            if found_file and os.path.exists(found_file):
                result_filename = f"localized_{int(time.time())}_{img_name}"
                dest_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
                shutil.copy2(found_file, dest_path)
                print(f"✅ YOLO localization successful: {result_filename}")
                return result_filename
            else:
                print("⚠️ YOLO output file not found")
                return None
        else:
            print("⚠️ No YOLO results returned")
            return None
            
    except Exception as e:
        print(f"❌ YOLO detection error: {str(e)}")
        return None

def save_scan(user_id, filename, result_filename, result, confidence, processing_time):
    """Save scan record to database"""
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
    """Get scans for a user"""
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
    """Get scan statistics for user"""
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    
    # Total scans
    c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ?", (user_id,))
    total = c.fetchone()[0]
    
    # Detected tumors
    c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result != 'No Tumor'", (user_id,))
    detected = c.fetchone()[0]
    
    # By type
    stats = {}
    for cls in CLASSES:
        c.execute("SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = ?", (user_id, cls))
        stats[cls] = c.fetchone()[0]
    
    conn.close()
    return total, detected, total - detected, stats

# ---------------- AUTH ROUTES ---------------- #
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if not all([name, email, password, confirm_password]):
            flash("All fields are required", "error")
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('register'))
        
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (name, email, password) VALUES (?,?,?)",
                (name, email, generate_password_hash(password))
            )
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists", "error")
        finally:
            conn.close()
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            session["user_email"] = user[2]
            flash(f"Welcome back, {user[1]}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "error")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "success")
    return redirect(url_for('login'))

# ---------------- MAIN ROUTES ---------------- #
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for('dashboard'))
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thank you for your message! We'll respond within 24 hours.", "success")
        return redirect(url_for('contact'))
    return render_template("contact.html")

@app.route("/features")
def features():
    return render_template("features.html")

# ---------------- DASHBOARD ROUTES ---------------- #
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    total, detected, healthy, stats = get_scan_stats(user_id)
    recent_scans = get_user_scans(user_id, limit=5)
    
    return render_template(
        "dashboard.html",
        user_name=session["user_name"],
        total_scans=total,
        detected=detected,
        healthy=healthy,
        stats=stats,
        recent_scans=recent_scans,
        classes=CLASSES
    )

@app.route("/history")
@login_required
def history():
    user_id = session["user_id"]
    scans = get_user_scans(user_id)
    return render_template("history.html", scans=scans, classes=CLASSES)

@app.route("/delete_scan/<int:scan_id>", methods=["POST"])
@login_required
def delete_scan(scan_id):
    user_id = session["user_id"]
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    
    # Get scan info first
    c.execute("SELECT filename, result_filename FROM scans WHERE id = ? AND user_id = ?", (scan_id, user_id))
    scan = c.fetchone()
    
    if scan:
        # Delete files
        try:
            filepath = os.path.join(UPLOAD_FOLDER, scan[0])
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if scan[1]:
                result_path = os.path.join(RESULT_FOLDER, scan[1])
                if os.path.exists(result_path):
                    os.remove(result_path)
        except Exception as e:
            print(f"Error deleting files: {e}")
        
        # Delete from database
        c.execute("DELETE FROM scans WHERE id = ? AND user_id = ?", (scan_id, user_id))
        conn.commit()
        flash("Scan deleted successfully", "success")
    else:
        flash("Scan not found", "error")
    
    conn.close()
    return redirect(url_for('history'))

# ---------------- PREDICTION ROUTES ---------------- #
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    start_time = time.time()
    
    if 'file' not in request.files:
        flash("No file selected", "error")
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for('dashboard'))
    
    if not allowed_file(file.filename):
        flash("File type not allowed. Please upload JPG, PNG, or JPEG files.", "error")
        return redirect(url_for('dashboard'))
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)
    
    # Predict with ResNet50
    try:
        if model:
            img = load_img(filepath, target_size=(128, 128))
            arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            preds = model.predict(arr, verbose=0)
            idx = np.argmax(preds[0])
            label = CLASSES[idx]
            confidence = float(preds[0][idx])
        else:
            # Mock prediction for testing
            idx = random.randint(0, 3)
            label = CLASSES[idx]
            confidence = random.uniform(0.75, 0.99)
        
        # Localization for tumors (ONLY YOLO, no heatmap)
        result_filename = None
        if label != "No Tumor":
            print(f"🔴 Tumor detected ({label}), running YOLO localization...")
            result_filename = detect_tumor_with_yolo(filepath)
        else:
            print("🟢 No tumor detected, skipping localization")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save to database
        user_id = session["user_id"]
        scan_id = save_scan(
            user_id, 
            unique_filename, 
            result_filename, 
            label, 
            confidence,
            processing_time
        )
        
        return render_template(
            "result.html",
            uploaded_img=unique_filename,
            result_img=result_filename,
            label=label,
            confidence=f"{confidence*100:.2f}%",
            scan_id=scan_id,
            processing_time=f"{processing_time:.2f}"
        )
        
    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "error")
        return redirect(url_for('dashboard'))

# ---------------- REPORT ROUTES ---------------- #
@app.route("/report_preview", methods=["POST"])
@login_required
def report_preview():
    label = request.form["label"]
    confidence = request.form["confidence"]
    image = request.form["image"]
    result_img = request.form.get("result_img")
    
    # Get tumor information
    tumor_info = TUMOR_INFO.get(label, "Analysis completed successfully.")
    
    if label == "No Tumor":
        summary = "✓ No tumor detected in the MRI scan. The brain structure appears normal."
        recommendations = "Regular checkups recommended as per standard medical guidelines."
        detailed_info = tumor_info
    else:
        summary = f"⚠️ AI analysis detected a {label} tumor with {confidence} confidence."
        recommendations = "Please consult with a neurosurgeon or radiologist for confirmation and treatment planning."
        detailed_info = tumor_info
    
    return render_template(
        "report.html",
        patient=session.get("user_name"),
        label=label,
        confidence=confidence,
        image=image,
        result_img=result_img,
        summary=summary,
        recommendations=recommendations,
        detailed_info=detailed_info,
        date=datetime.now().strftime("%d %B %Y at %H:%M"),
        report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

@app.route("/download_report", methods=["POST"])
@login_required
def download_report():
    label = request.form["label"]
    confidence = request.form["confidence"]
    image = request.form["image"]
    result_img = request.form.get("result_img")
    
    pdf_name = f"NeuroScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)
    
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#8b5cf6'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("NeuroScan AI Medical Report", title_style))
    
    # Report ID and Date
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    story.append(Paragraph(f"Report ID: RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}", info_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y at %H:%M')}", info_style))
    story.append(Spacer(1, 20))
    
    # Patient Info
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Paragraph(f"Name: {session.get('user_name')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Results
    story.append(Paragraph("Analysis Results", styles['Heading2']))
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#ef4444') if label != "No Tumor" else colors.HexColor('#10b981'),
        spaceAfter=10
    )
    story.append(Paragraph(f"Diagnosis: {label}", result_style))
    story.append(Paragraph(f"Confidence: {confidence}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Tumor Information
    story.append(Paragraph("Clinical Information", styles['Heading3']))
    tumor_info = TUMOR_INFO.get(label, "Analysis completed successfully.")
    story.append(Paragraph(tumor_info, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Original image
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], image)
    if os.path.exists(original_path):
        story.append(Paragraph("Original MRI Scan", styles['Heading3']))
        img = RLImage(original_path, width=4*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
    # Result image
    if result_img:
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_img)
        if os.path.exists(result_path):
            story.append(Paragraph("Tumor Localization (YOLOv8 Detection)", styles['Heading3']))
            img = RLImage(result_path, width=4*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("Clinical Summary", styles['Heading2']))
    if label == "No Tumor":
        story.append(Paragraph("✓ No tumor detected. Regular monitoring recommended.", styles['Normal']))
    else:
        story.append(Paragraph(f"⚠️ {label} tumor detected. Immediate consultation with a specialist is recommended.", styles['Normal']))
    
    # Recommendations
    if label != "No Tumor":
        story.append(Spacer(1, 10))
        story.append(Paragraph("Recommendations", styles['Heading3']))
        story.append(Paragraph("• Consult with a neurosurgeon within 1-2 weeks", styles['Normal']))
        story.append(Paragraph("• Bring this report and original MRI to your appointment", styles['Normal']))
        story.append(Paragraph("• Further imaging may be required for confirmation", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=1
    )
    story.append(Paragraph("This report is generated by AI and should be reviewed by a qualified healthcare professional.", footer_style))
    story.append(Paragraph("© 2024 NeuroScan AI. All rights reserved.", footer_style))
    
    doc.build(story)
    
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        pdf_name,
        as_attachment=True,
        download_name=pdf_name
    )

# ---------------- FILE SERVING ---------------- #
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/results/<filename>')
@login_required
def result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

# ---------------- CLEANUP ROUTE ---------------- #
@app.route('/cleanup_yolo', methods=['POST'])
@login_required
def cleanup_yolo():
    """Clean up YOLO temporary files"""
    try:
        runs_dir = 'runs/detect'
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir)
            os.makedirs(runs_dir, exist_ok=True)
        flash("Temporary files cleaned up", "success")
    except Exception as e:
        flash(f"Error cleaning up: {e}", "error")
    return redirect(url_for('dashboard'))

# ---------------- CONTEXT PROCESSOR ---------------- #
@app.context_processor
def utility_processor():
    return dict(now=datetime.now)

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    # Delete old database if it exists to avoid schema conflicts
    if os.path.exists("database.db"):
        os.remove("database.db")
        print("🗑️ Removed old database")
    
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
