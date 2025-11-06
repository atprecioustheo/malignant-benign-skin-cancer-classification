from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    redirect,
    url_for,
    session,
    flash,
)
import numpy as np
import cv2
import os
import pickle
from werkzeug.utils import secure_filename
import logging
from functools import lru_cache, wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key_here"  # Change this in production
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
db = SQLAlchemy(app)


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    display_name = db.Column(db.String(120))
    bio = db.Column(db.Text)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    # Settings
    email_notifications = db.Column(db.Boolean, default=True)
    analysis_reminders = db.Column(db.Boolean, default=False)
    system_updates = db.Column(db.Boolean, default=True)
    data_sharing = db.Column(db.Boolean, default=False)
    analytics = db.Column(db.Boolean, default=True)


class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(120))
    notes = db.Column(db.Text)
    prediction = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(500))


# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
    logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


# Initialize DB if not present
@app.before_request
def create_tables():
    db.create_all()


# Main routes
@app.route("/")
@login_required
def home():
    user = User.query.get(session["user_id"])
    return render_template("index.html", user=user)


@app.route("/upload")
@login_required
def upload_page():
    user = User.query.get(session["user_id"])
    return render_template("upload.html", user=user)


@app.route("/history")
@login_required
def history_page():
    user = User.query.get(session["user_id"])
    return render_template("history.html", user=user)


@app.route("/profile")
@login_required
def profile_page():
    user = User.query.get(session["user_id"])
    return render_template("profile.html", user=user)


@app.route("/settings")
@login_required
def settings_page():
    user = User.query.get(session["user_id"])
    return render_template("settings.html", user=user)


# User registration route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("signup"))
        hashed_pw = generate_password_hash(password)
        user = User(email=email, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        session["user_id"] = user.id
        return redirect(url_for("home"))
    return render_template("signup.html")


# User login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for("home"))
        flash("Invalid credentials.")
        return redirect(url_for("login"))
    return render_template("login.html")


# User logout
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


# API Routes
@app.route("/api/dashboard")
@login_required
def api_dashboard():
    user_id = session.get("user_id")
    total = ImageRecord.query.filter_by(user_id=user_id).count()
    benign = ImageRecord.query.filter_by(user_id=user_id, prediction="Benign").count()
    malignant = ImageRecord.query.filter_by(
        user_id=user_id, prediction="Malignant"
    ).count()
    return jsonify({"total": total, "benign": benign, "malignant": malignant})


@app.route("/api/history")
@login_required
def api_history():
    user_id = session.get("user_id")
    history = (
        ImageRecord.query.filter_by(user_id=user_id)
        .order_by(ImageRecord.date.desc())
        .all()
    )

    history_data = []
    for record in history:
        history_data.append(
            {
                "id": record.id,
                "name": record.name or f"Image {record.id}",
                "prediction": record.prediction,
                "confidence": record.confidence or 95.0,
                "date": record.date.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": record.notes,
            }
        )

    return jsonify({"history": history_data})


@app.route("/api/profile", methods=["GET"])
@login_required
def api_get_profile():
    user = User.query.get(session["user_id"])
    total_analyses = ImageRecord.query.filter_by(user_id=user.id).count()

    return jsonify(
        {
            "id": user.id,
            "name": user.display_name or user.email.split("@")[0],
            "email": user.email,
            "username": user.email.split("@")[0],
            "member_since": user.created_at.strftime("%B %Y")
            if user.created_at
            else "January 2024",
            "total_analyses": total_analyses,
            "last_login": "Today",
            "bio": user.bio or "",
        }
    )


@app.route("/api/update-profile", methods=["POST"])
@login_required
def api_update_profile():
    user = User.query.get(session["user_id"])
    data = request.get_json()

    if data.get("displayName"):
        user.display_name = data["displayName"]
    if data.get("bio") is not None:
        user.bio = data["bio"]

    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/change-password", methods=["POST"])
@login_required
def api_change_password():
    user = User.query.get(session["user_id"])
    data = request.get_json()

    if not check_password_hash(user.password, data["currentPassword"]):
        return jsonify({"error": "Current password is incorrect"}), 400

    user.password = generate_password_hash(data["newPassword"])
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/update-settings", methods=["POST"])
@login_required
def api_update_settings():
    user = User.query.get(session["user_id"])
    data = request.get_json()

    for key, value in data.items():
        if hasattr(user, key):
            setattr(user, key, value)

    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/delete-account", methods=["DELETE"])
@login_required
def api_delete_account():
    user = User.query.get(session["user_id"])

    # Delete all user's image records
    ImageRecord.query.filter_by(user_id=user.id).delete()

    # Delete user
    db.session.delete(user)
    db.session.commit()

    session.pop("user_id", None)
    return jsonify({"success": True})


@app.route("/api/clear-data", methods=["DELETE"])
@login_required
def api_clear_data():
    user_id = session.get("user_id")

    # Delete all user's image records
    ImageRecord.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    return jsonify({"success": True})


@app.route("/api/analysis/<int:analysis_id>", methods=["GET"])
@login_required
def api_get_analysis(analysis_id):
    user_id = session.get("user_id")
    record = ImageRecord.query.filter_by(id=analysis_id, user_id=user_id).first()

    if not record:
        return jsonify({"error": "Analysis not found"}), 404

    # Extract probabilities from notes if available
    benign_prob = 0.0
    malignant_prob = 0.0
    if record.notes and "Benign:" in record.notes:
        try:
            parts = record.notes.split(", ")
            benign_prob = float(parts[0].split(": ")[1].replace("%", ""))
            malignant_prob = float(parts[1].split(": ")[1].replace("%", ""))
        except:
            pass

    return jsonify(
        {
            "id": record.id,
            "name": record.name or f"Image {record.id}",
            "prediction": record.prediction,
            "confidence": record.confidence or 95.0,
            "date": record.date.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": record.notes,
            "image_path": record.image_path,
            "benign_probability": benign_prob,
            "malignant_probability": malignant_prob,
        }
    )


@app.route("/api/analysis/<int:analysis_id>", methods=["DELETE"])
@login_required
def api_delete_analysis(analysis_id):
    user_id = session.get("user_id")
    record = ImageRecord.query.filter_by(id=analysis_id, user_id=user_id).first()

    if not record:
        return jsonify({"error": "Analysis not found"}), 404

    db.session.delete(record)
    db.session.commit()

    return jsonify({"success": True})


# Global variables for model and preprocessing components
model = None
scaler = None
encoder = None
pca = None


# Load model and preprocessing components
def load_model_and_preprocessors():
    global model, scaler, encoder, pca
    if model is not None:
        return model, scaler, encoder, pca

    base_path = os.path.dirname(__file__)

    try:
        import joblib

        # Load model
        model_path = os.path.join(base_path, "svm_model.pkl")
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded svm_model.pkl")

        # Load preprocessing components
        scaler_path = os.path.join(base_path, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        logger.info(f"✅ Loaded scaler.pkl")

        encoder_path = os.path.join(base_path, "encoder.pkl")
        encoder = joblib.load(encoder_path)
        logger.info(f"✅ Loaded encoder.pkl")

        pca_path = os.path.join(base_path, "pca.pkl")
        pca = joblib.load(pca_path)
        logger.info(f"✅ Loaded pca.pkl")

        # Test with dummy data
        dummy_data = np.random.rand(1, 250).astype(np.float32)
        test_pred = model.predict(dummy_data)
        logger.info(f"Model test successful - Prediction: {test_pred}")

        return model, scaler, encoder, pca

    except Exception as e:
        logger.error(f"Error loading model/preprocessors: {str(e)}")
        model = scaler = encoder = pca = None
        raise


# Initialize model and preprocessors at startup
try:
    model, scaler, encoder, pca = load_model_and_preprocessors()
    logger.info("Model and preprocessors initialized successfully at startup")
except Exception as e:
    logger.error(f"Failed to initialize model/preprocessors at startup: {str(e)}")
    model = scaler = encoder = pca = None


# Preprocess image for prediction - matches training preprocessing exactly
def preprocess_image(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")

        logger.info(f"Original image shape: {img.shape}")

        # Resize to 64x64 (same as training)
        img_resized = cv2.resize(img, (64, 64))

        # Convert to grayscale (same as training)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Flatten to 1D array (same as training)
        img_flattened = img_gray.reshape(1, -1)

        # Apply StandardScaler (same as training)
        img_scaled = scaler.transform(img_flattened)

        # Apply PCA (same as training)
        img_pca = pca.transform(img_scaled)

        logger.info(f"Preprocessed image shape: {img_pca.shape}")
        logger.info(f"Feature range: min={img_pca.min():.3f}, max={img_pca.max():.3f}")

        return img_pca

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


# Prediction endpoint
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    global model, scaler, encoder, pca

    # Force reload of all components to ensure they're available
    try:
        model, scaler, encoder, pca = load_model_and_preprocessors()
        logger.info("Components reloaded for prediction")
    except Exception as e:
        logger.error(f"Failed to load components: {str(e)}")
        return jsonify({"error": "Model components not available"}), 503

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        # Preprocess and predict
        processed_image = preprocess_image(filepath)

        # Debug: Log processed image stats
        logger.info(
            f"Processed image stats: min={processed_image.min():.3f}, max={processed_image.max():.3f}, mean={processed_image.mean():.3f}"
        )

        prediction = model.predict(processed_image)[0]
        logger.info(f"Raw prediction: {prediction} (type: {type(prediction)})")

        # Get prediction probabilities if available
        confidence = 85.0  # Default confidence for balanced model
        benign_prob = 0.0
        malignant_prob = 0.0

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(processed_image)[0]
            benign_prob = float(proba[0] * 100)
            malignant_prob = float(proba[1] * 100)

            # Adjusted threshold: classify as malignant if probability > 35% (reduces false negatives)
            adjusted_prediction = 1 if malignant_prob > 35.0 else 0

            # Calculate confidence based on probability spread
            prob_diff = abs(benign_prob - malignant_prob)
            if prob_diff > 30:
                confidence = min(95.0, 70.0 + prob_diff)
            elif prob_diff > 15:
                confidence = min(85.0, 60.0 + prob_diff)
            else:
                confidence = max(55.0, 50.0 + prob_diff)

            logger.info(
                f"Probabilities: Benign={proba[0]:.3f}, Malignant={proba[1]:.3f}"
            )
            logger.info(f"Original: {prediction}, Adjusted: {adjusted_prediction}")
        else:
            adjusted_prediction = prediction
            benign_prob = 50.0
            malignant_prob = 50.0
            confidence = 75.0

        # Use adjusted prediction for medical safety
        result = "Malignant" if adjusted_prediction == 1 else "Benign"

        logger.info(f"Final result: {result}")
        logger.info(f"Final confidence: {confidence:.1f}%")

        # Save to database
        user_id = session.get("user_id")
        record = ImageRecord(
            user_id=user_id,
            name=filename,
            prediction=result,
            confidence=confidence,
            image_path=filepath,
            notes=f"Benign: {benign_prob:.1f}%, Malignant: {malignant_prob:.1f}%",
        )
        db.session.add(record)
        db.session.commit()

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(
            {
                "prediction": result,
                "confidence": confidence,
                "benign_probability": benign_prob,
                "malignant_probability": malignant_prob,
                "message": f"Prediction: {result} (Confidence: {confidence:.1f}%)",
                "disclaimer": "⚠️ MEDICAL DISCLAIMER: This is an AI screening tool, not a medical diagnosis. Always consult a healthcare professional.",
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Health check endpoint
@app.route("/health")
def health_check():
    global model, scaler, encoder, pca
    try:
        if model is None or scaler is None or pca is None:
            model, scaler, encoder, pca = load_model_and_preprocessors()

        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "model_type": str(type(model)),
                "preprocessors_loaded": True,
                "message": "Balanced SVM model with preprocessing pipeline is ready",
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "unhealthy", "model_loaded": False, "error": str(e)}
        ), 500


if __name__ == "__main__":
    try:
        # Test model and preprocessors loading on startup
        test_model, test_scaler, test_encoder, test_pca = load_model_and_preprocessors()
        logger.info("✅ Model and preprocessors loaded successfully on startup")

        # Run the Flask app
        app.run(debug=True, host="0.0.0.0", port=5000)

    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print(
            "Make sure svm_model.pkl, scaler.pkl, encoder.pkl, and pca.pkl exist in the same directory as app.py"
        )
        # Still run the app even if model fails to load initially
        app.run(debug=True, host="0.0.0.0", port=5000)
