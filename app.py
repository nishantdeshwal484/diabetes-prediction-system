import gradio as gr
import pickle
import numpy as np

# ============================================================
# Diabetes Prediction System
# ============================================================

# Load the trained KNN model
try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(
        "Model file 'diabetes_model.pkl' not found. "
        "Run train_model.py first."
    )


def predict_diabetes(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    """Predict diabetes risk using the trained KNN pipeline."""
    input_features = np.array([[
        pregnancies, glucose, bp, skin_thickness,
        insulin, bmi, dpf, age
    ]])

    prediction = model.predict(input_features)

    if prediction[0] == 1:
        return """
        <div class="result-card risk">
            <div class="result-icon">⚠️</div>
            <div>
                <div class="result-title">Diabetic / Higher Risk Prediction</div>
                <div class="result-text">
                    The trained KNN model classified this input as <b>1 (Diabetic)</b>.
                    Please treat this as a machine-learning prediction, not a medical diagnosis.
                </div>
            </div>
        </div>
        """
    else:
        return """
        <div class="result-card safe">
            <div class="result-icon">✓</div>
            <div>
                <div class="result-title">Non-Diabetic / Lower Risk Prediction</div>
                <div class="result-text">
                    The trained KNN model classified this input as <b>0 (Non-Diabetic)</b>.
                    Please treat this as a machine-learning prediction, not a medical diagnosis.
                </div>
            </div>
        </div>
        """


# ============================================================
# Custom styling
# ============================================================

css = r"""
/* ---------- Global ---------- */
:root {
    --ink: #12202f;
    --muted: #617184;
    --line: #dce7ee;
    --teal: #0f8b8d;
    --teal-dark: #086568;
    --blue: #2563eb;
    --soft: #f4fafb;
    --white: #ffffff;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    background: #f7fbfc !important;
    color: var(--ink) !important;
}

body {
    background: #f7fbfc !important;
}

/* ---------- Hero ---------- */
.hero {
    margin: 18px 0 22px;
    padding: 42px 42px 38px;
    border-radius: 28px;
    color: white;
    background:
        radial-gradient(circle at 85% 20%, rgba(255,255,255,.16), transparent 26%),
        linear-gradient(135deg, #073b4c 0%, #0b6f73 52%, #138a8a 100%);
    box-shadow: 0 18px 45px rgba(8, 74, 82, .18);
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 50%;
    right: -65px;
    bottom: -100px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 12px;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    font-size: 13px;
    letter-spacing: .3px;
    margin-bottom: 13px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 56px);
    line-height: 1.03;
    margin: 0 0 12px;
    letter-spacing: -1.8px;
}

.hero p {
    max-width: 800px;
    margin: 0;
    font-size: 17px;
    line-height: 1.65;
    color: rgba(255,255,255,.88);
}

/* ---------- Stat cards ---------- */
.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 13px;
    margin: 0 0 22px;
}

.stat {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 17px;
    box-shadow: 0 7px 22px rgba(19, 53, 66, .05);
}

.stat .number {
    font-size: 25px;
    font-weight: 800;
    color: var(--teal-dark);
}

.stat .label {
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
}

/* ---------- Section cards ---------- */
.section-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 25px;
    margin: 4px 0 18px;
    box-shadow: 0 7px 24px rgba(19, 53, 66, .045);
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 6px;
}

.section-subtitle {
    color: var(--muted);
    margin-bottom: 20px;
    line-height: 1.55;
}

/* ---------- Predictor ---------- */
.predictor-head {
    padding: 8px 2px 16px;
}

.predictor-head h2 {
    margin: 0 0 7px;
    font-size: 29px;
}

.predictor-head p {
    margin: 0;
    color: var(--muted);
}

.input-label {
    font-weight: 700 !important;
}

.primary-btn {
    border-radius: 14px !important;
    min-height: 50px !important;
    font-size: 16px !important;
    font-weight: 750 !important;
}

.result-card {
    display: flex;
    gap: 15px;
    align-items: flex-start;
    border-radius: 17px;
    padding: 18px;
    margin-top: 7px;
    border: 1px solid;
}

.result-card.risk {
    background: #fff7ed;
    border-color: #fed7aa;
}

.result-card.safe {
    background: #ecfdf5;
    border-color: #a7f3d0;
}

.result-icon {
    font-size: 28px;
    line-height: 1;
}

.result-title {
    font-weight: 800;
    font-size: 17px;
    margin-bottom: 5px;
}

.result-text {
    color: #526272;
    line-height: 1.5;
    font-size: 14px;
}

/* ---------- Info grid ---------- */
.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
}

.info-box {
    border: 1px solid var(--line);
    background: #fbfdfe;
    border-radius: 16px;
    padding: 18px;
}

.info-box h3 {
    margin: 0 0 7px;
    font-size: 16px;
}

.info-box p {
    margin: 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 14px;
}

/* ---------- Feature pills ---------- */
.feature-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
}

.pill {
    padding: 9px 12px;
    border-radius: 999px;
    background: #edf8f8;
    border: 1px solid #cdebec;
    color: #075f62;
    font-size: 13px;
    font-weight: 650;
}

/* ---------- Workflow ---------- */
.workflow {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.step {
    padding: 17px;
    border-radius: 16px;
    background: #f7fbfc;
    border: 1px solid var(--line);
}

.step-no {
    display: inline-flex;
    width: 30px;
    height: 30px;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #dff4f4;
    color: #075f62;
    font-weight: 800;
    margin-bottom: 10px;
}

.step h3 {
    margin: 0 0 5px;
    font-size: 15px;
}

.step p {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

/* ---------- Disclaimer ---------- */
.disclaimer {
    background: #fffaf0;
    border: 1px solid #f6dfab;
    border-radius: 17px;
    padding: 16px 18px;
    color: #6d5724;
    line-height: 1.55;
    font-size: 13px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #718091;
    padding: 25px 10px 35px;
    font-size: 13px;
}

/* ---------- Mobile ---------- */
@media (max-width: 800px) {
    .stats,
    .workflow,
    .info-grid {
        grid-template-columns: 1fr 1fr;
    }

    .hero {
        padding: 30px 24px;
    }
}

@media (max-width: 560px) {
    .stats,
    .workflow,
    .info-grid {
        grid-template-columns: 1fr;
    }
}
"""

# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Diabetes Prediction System",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue"
    ),
    css=css
) as app:

    gr.HTML("""
    <section class="hero">
        <div class="hero-badge">🩺 Machine Learning • K-Nearest Neighbors</div>
        <h1>Diabetes Prediction<br>System</h1>
        <p>
            An interactive machine-learning application that uses eight patient
            health measurements to generate a diabetes classification through
            a trained K-Nearest Neighbors (KNN) model.
        </p>
    </section>

    <div class="stats">
        <div class="stat"><div class="number">KNN</div><div class="label">Machine Learning Algorithm</div></div>
        <div class="stat"><div class="number">8</div><div class="label">Input Health Features</div></div>
        <div class="stat"><div class="number">80/20</div><div class="label">Train / Test Split</div></div>
        <div class="stat"><div class="number">2</div><div class="label">Prediction Classes</div></div>
    </div>
    """)

    with gr.Tabs():

        # ---------------- Prediction ----------------
        with gr.Tab("🔬 Predict"):
            gr.HTML("""
            <div class="section-card">
                <div class="predictor-head">
                    <h2>Patient Prediction Panel</h2>
                    <p>
                        Enter the patient measurements below and click
                        <b>Predict Status</b>. The model returns one of two
                        classes: Diabetic or Non-Diabetic.
                    </p>
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    pregnancies = gr.Number(
                        label="Pregnancies",
                        minimum=0,
                        step=1,
                        value=0
                    )
                    glucose = gr.Number(
                        label="Glucose Level",
                        minimum=0,
                        value=120
                    )
                    bp = gr.Number(
                        label="Blood Pressure (mm Hg)",
                        minimum=0,
                        value=70
                    )
                    skin_thickness = gr.Number(
                        label="Skin Thickness (mm)",
                        minimum=0,
                        value=20
                    )

                with gr.Column():
                    insulin = gr.Number(
                        label="Insulin (mu U/ml)",
                        minimum=0,
                        value=79
                    )
                    bmi = gr.Number(
                        label="BMI",
                        minimum=0,
                        value=25.0
                    )
                    dpf = gr.Number(
                        label="Diabetes Pedigree Function",
                        minimum=0,
                        value=0.5
                    )
                    age = gr.Number(
                        label="Age (years)",
                        minimum=1,
                        step=1,
                        value=30
                    )

            submit_btn = gr.Button(
                "🔍 Predict Status",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            output = gr.HTML(
                label="Prediction Result",
                value="""
                <div class="section-card">
                    <div class="section-subtitle">
                        Your prediction will appear here after you submit the patient information.
                    </div>
                </div>
                """
            )

            submit_btn.click(
                fn=predict_diabetes,
                inputs=[
                    pregnancies, glucose, bp, skin_thickness,
                    insulin, bmi, dpf, age
                ],
                outputs=output
            )

            gr.HTML("""
            <div class="disclaimer">
                <b>⚕️ Important:</b> This application is an educational machine-learning
                project. Its output is a model prediction and should not be used as
                a medical diagnosis or as a substitute for professional medical advice.
            </div>
            """)

        # ---------------- About project ----------------
        with gr.Tab("📘 About Project"):
            gr.HTML("""
            <div class="section-card">
                <div class="section-title">Project Overview</div>
                <div class="section-subtitle">
                    This project demonstrates how a supervised machine-learning model
                    can be connected to an interactive web interface. The application
                    takes patient measurements as inputs and uses a trained KNN
                    classifier to produce a binary prediction.
                </div>

                <div class="info-grid">
                    <div class="info-box">
                        <h3>🎯 Objective</h3>
                        <p>
                            Build an accessible web application that demonstrates
                            diabetes prediction using a trained K-Nearest Neighbors model.
                        </p>
                    </div>
                    <div class="info-box">
                        <h3>🧠 Algorithm</h3>
                        <p>
                            K-Nearest Neighbors (KNN) classifies a new input according
                            to nearby examples in the training data.
                        </p>
                    </div>
                    <div class="info-box">
                        <h3>⚙️ Preprocessing</h3>
                        <p>
                            A StandardScaler is included in the machine-learning
                            pipeline before the KNN classifier so that features are
                            placed on comparable scales.
                        </p>
                    </div>
                    <div class="info-box">
                        <h3>🌐 Interface</h3>
                        <p>
                            Gradio provides the interactive browser-based interface,
                            allowing users to enter measurements without writing code.
                        </p>
                    </div>
                </div>
            </div>

            <div class="section-card">
                <div class="section-title">Technology Stack</div>
                <div class="section-subtitle">The main technologies used in this project.</div>
                <div class="feature-row">
                    <span class="pill">🐍 Python</span>
                    <span class="pill">🤖 Scikit-learn</span>
                    <span class="pill">📊 Pandas</span>
                    <span class="pill">🔢 NumPy</span>
                    <span class="pill">🖥️ Gradio</span>
                    <span class="pill">📦 Pickle</span>
                    <span class="pill">🧠 KNN</span>
                    <span class="pill">📏 StandardScaler</span>
                </div>
            </div>
            """)

        # ---------------- Methodology ----------------
        with gr.Tab("🧪 Methodology"):
            gr.HTML("""
            <div class="section-card">
                <div class="section-title">How the System Works</div>
                <div class="section-subtitle">
                    The complete project follows a simple machine-learning workflow.
                </div>

                <div class="workflow">
                    <div class="step">
                        <div class="step-no">1</div>
                        <h3>Load Dataset</h3>
                        <p>Read the diabetes dataset from <b>diabetes.csv</b>.</p>
                    </div>
                    <div class="step">
                        <div class="step-no">2</div>
                        <h3>Prepare Data</h3>
                        <p>Separate the input features from the <b>Outcome</b> target.</p>
                    </div>
                    <div class="step">
                        <div class="step-no">3</div>
                        <h3>Train KNN</h3>
                        <p>Scale features and train KNN with <b>5 neighbors</b>.</p>
                    </div>
                    <div class="step">
                        <div class="step-no">4</div>
                        <h3>Predict</h3>
                        <p>Send new patient measurements through the saved pipeline.</p>
                    </div>
                </div>
            </div>

            <div class="section-card">
                <div class="section-title">Input Features</div>
                <div class="section-subtitle">
                    These are the eight values collected by the prediction form.
                </div>

                <div class="info-grid">
                    <div class="info-box">
                        <h3>Pregnancies</h3>
                        <p>Number of pregnancies recorded for the patient.</p>
                    </div>
                    <div class="info-box">
                        <h3>Glucose</h3>
                        <p>Glucose measurement supplied to the model.</p>
                    </div>
                    <div class="info-box">
                        <h3>Blood Pressure</h3>
                        <p>Blood pressure measurement in mm Hg.</p>
                    </div>
                    <div class="info-box">
                        <h3>Skin Thickness</h3>
                        <p>Skin thickness measurement in millimetres.</p>
                    </div>
                    <div class="info-box">
                        <h3>Insulin</h3>
                        <p>Insulin measurement supplied to the model.</p>
                    </div>
                    <div class="info-box">
                        <h3>BMI</h3>
                        <p>Body mass index value.</p>
                    </div>
                    <div class="info-box">
                        <h3>Diabetes Pedigree Function</h3>
                        <p>The DPF feature used by the trained model.</p>
                    </div>
                    <div class="info-box">
                        <h3>Age</h3>
                        <p>Patient age in years.</p>
                    </div>
                </div>
            </div>
            """)

        # ---------------- Project files ----------------
        with gr.Tab("📂 Project Guide"):
            gr.HTML("""
            <div class="section-card">
                <div class="section-title">Project File Structure</div>
                <div class="section-subtitle">
                    Keep these files together when running the application locally.
                </div>

                <div class="info-grid">
                    <div class="info-box">
                        <h3>app.py</h3>
                        <p>The main Gradio web application and prediction interface.</p>
                    </div>
                    <div class="info-box">
                        <h3>train_model.py</h3>
                        <p>Loads the dataset, trains the KNN pipeline and saves the model.</p>
                    </div>
                    <div class="info-box">
                        <h3>diabetes.csv</h3>
                        <p>The dataset used by the training script.</p>
                    </div>
                    <div class="info-box">
                        <h3>diabetes_model.pkl</h3>
                        <p>The saved trained model loaded by the web application.</p>
                    </div>
                    <div class="info-box">
                        <h3>requirements.txt</h3>
                        <p>Python package requirements for the project environment.</p>
                    </div>
                    <div class="info-box">
                        <h3>README.md</h3>
                        <p>Project documentation and setup information.</p>
                    </div>
                </div>
            </div>

            <div class="section-card">
                <div class="section-title">Run the Project</div>
                <div class="section-subtitle">
                    After installing the required packages and activating your virtual
                    environment, run the following commands from the project directory.
                </div>

                <pre style="background:#10212b;color:#e7f7f8;padding:18px;border-radius:14px;overflow:auto;">
python train_model.py
python app.py
                </pre>

                <p style="color:#617184;line-height:1.6;">
                    If <b>diabetes_model.pkl</b> already exists and is compatible with
                    your installed scikit-learn version, you can normally run
                    <b>python app.py</b> directly.
                </p>
            </div>

            <div class="disclaimer">
                <b>Project scope:</b> This interface is designed to present and demonstrate
                the supplied machine-learning project. It does not establish clinical
                accuracy, and predictions should not be interpreted as a diagnosis.
            </div>
            """)

    gr.HTML("""
    <div class="footer">
        🩺 Diabetes Prediction System &nbsp;•&nbsp;
        KNN Machine Learning Demo &nbsp;•&nbsp;
        Built with Python + Scikit-learn + Gradio
    </div>
    """)


if __name__ == "__main__":
    app.launch()