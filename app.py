from pathlib import Path
import pickle
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_INFO = {
    "rice": ("🌾", "Rice generally grows in warm conditions with plenty of water.", ["Maintain suitable soil moisture.", "Check drainage and water availability.", "Use balanced nutrients based on a soil test."]),
    "maize": ("🌽", "Maize needs warm weather, sunlight, and well-drained soil.", ["Avoid waterlogging.", "Control weeds.", "Provide adequate nutrients."]),
    "chickpea": ("🫘", "Chickpea is a pulse crop that generally prefers cool, dry conditions.", ["Avoid excessive irrigation.", "Use well-drained soil.", "Monitor pests and disease."]),
    "kidneybeans": ("🫘", "Kidney beans need suitable warmth and well-drained soil.", ["Avoid waterlogged soil.", "Use healthy seeds.", "Inspect plants regularly."]),
    "pigeonpeas": ("🌱", "Pigeon pea is a pulse crop that can tolerate some dry conditions once established.", ["Ensure good drainage.", "Avoid overwatering.", "Use locally recommended varieties."]),
    "mothbeans": ("🌱", "Moth bean is commonly grown in warm, relatively dry regions.", ["Avoid excess irrigation.", "Control weeds early.", "Follow local sowing advice."]),
    "mungbean": ("🌱", "Mung bean is a short-duration pulse crop suited to warm conditions.", ["Avoid standing water.", "Use healthy seeds.", "Monitor pod formation."]),
    "blackgram": ("🌱", "Black gram generally needs warm conditions and good drainage.", ["Prevent waterlogging.", "Monitor pests.", "Use balanced nutrient management."]),
    "lentil": ("🌱", "Lentil is generally grown as a cool-season pulse crop.", ["Avoid excess moisture.", "Use well-drained soil.", "Follow local sowing dates."]),
    "pomegranate": ("🍎", "Pomegranate is a fruit crop that prefers sunny conditions and well-drained soil.", ["Avoid waterlogging.", "Manage the orchard appropriately.", "Use stage-appropriate irrigation."]),
    "banana": ("🍌", "Banana generally needs warm conditions, moisture, and nutrient-rich soil.", ["Maintain consistent moisture.", "Use soil-test-based nutrients.", "Protect plants from strong winds."]),
    "mango": ("🥭", "Mango is a fruit tree suited to warm climates.", ["Avoid waterlogging.", "Manage orchard pruning.", "Monitor fruit development."]),
    "grapes": ("🍇", "Grapes need suitable sunlight, drainage, and variety-specific care.", ["Use suitable support.", "Manage canopy growth.", "Monitor disease risk."]),
    "watermelon": ("🍉", "Watermelon generally grows best in warm conditions and well-drained soil.", ["Avoid overwatering near harvest.", "Allow space for vines.", "Monitor fruit development."]),
    "muskmelon": ("🍈", "Muskmelon is a warm-season crop that prefers well-drained soil.", ["Avoid standing water.", "Provide sunlight.", "Water according to crop stage."]),
    "apple": ("🍎", "Apple is a temperate fruit crop with needs that depend on the variety.", ["Choose a locally suitable variety.", "Manage pruning.", "Follow local disease-management advice."]),
    "orange": ("🍊", "Orange is a citrus crop needing suitable warmth and well-drained soil.", ["Avoid waterlogging.", "Monitor nutrient status.", "Use local orchard practices."]),
    "papaya": ("🍈", "Papaya generally prefers warm weather and well-drained soil.", ["Protect roots from waterlogging.", "Monitor disease.", "Use suitable varieties."]),
    "coconut": ("🥥", "Coconut is a tropical crop that generally needs warmth and adequate moisture.", ["Maintain suitable moisture.", "Use local nutrient recommendations.", "Manage the area around palms."]),
    "cotton": ("☁️", "Cotton generally needs warm weather and careful water and pest management.", ["Scout for pests.", "Avoid excessive nitrogen.", "Use integrated pest management."]),
    "jute": ("🌿", "Jute is a fibre crop commonly grown in warm, humid conditions.", ["Maintain suitable moisture.", "Control weeds early.", "Follow local harvest guidance."]),
    "coffee": ("☕", "Coffee is a perennial crop whose needs depend on species and local conditions.", ["Use suitable varieties.", "Manage shade and drainage.", "Monitor pests and disease."]),
}

st.set_page_config(page_title="Smart Crop | Recommendation", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 80% 0%,#123629 0%,#0b1511 38%,#080d0b 100%);color:#edf7ef}
[data-testid="stSidebar"]{background:#0d1c16;border-right:1px solid #254638}
.block-container{padding-top:1.8rem;max-width:1400px}
.hero{padding:1.5rem 1.7rem;border:1px solid #285b43;border-radius:22px;background:linear-gradient(115deg,#18442f,#0c1d16);margin-bottom:1.25rem}
.eyebrow{color:#9fe2b7;font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700}
.hero h1{margin:.35rem 0;color:#f3fff5;font-size:2.2rem}.hero p,.muted{color:#c2d9ca}
.panel,.result-card{border:1px solid #2b7952;border-radius:18px;padding:1.25rem;background:linear-gradient(135deg,#123e2b,#10271d)}
.result-label{color:#9fe2b7;font-size:.8rem;text-transform:uppercase;letter-spacing:.12em}
.result-name{color:#f1fff5;font-size:2rem;font-weight:800;margin:.25rem 0}
div.stButton>button{background:linear-gradient(90deg,#2c9b62,#48bb78);color:#07130c;border:0;border-radius:11px;min-height:3rem;font-weight:800}
div[data-testid="stMetric"]{background:#10251b;border:1px solid #254638;padding:1rem;border-radius:14px}
div[data-testid="stMetricLabel"]{color:#b5c9bc} div[data-testid="stMetricValue"]{color:#f2fff5}
</style>
""", unsafe_allow_html=True)

def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file nahi mili: {MODEL_PATH.name}. Isse project ke main folder mein rakho."
    try:
        try:
            return joblib.load(MODEL_PATH), None
        except Exception:
            with MODEL_PATH.open("rb") as f:
                return pickle.load(f), None
    except Exception as e:
        return None, f"Model load nahi ho paya: {e}"

model, model_error = load_model()

with st.sidebar:
    st.markdown("## 🌱 Smart Crop")
    st.caption("PRECISION FARMING • MACHINE LEARNING")
    st.markdown("---")
    st.markdown("**01**  Crop recommendation")
    st.markdown("**02**  Soil analytics")
    st.markdown("**03**  Crop guide")
    st.markdown("---")
    st.caption("Educational ML project. Verify recommendations with local agricultural advice.")

st.markdown("""
<div class="hero">
<div class="eyebrow">Smart agriculture / ML-powered insights</div>
<h1>Grow smarter. Choose better.</h1>
<p>Enter soil nutrients and weather conditions to get a crop recommendation from your trained model.</p>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(model_error)
    st.info("Expected structure: app.py, crop_model.pkl, and optionally data/Crop_recommendation.csv in the project root.")

st.markdown("### Soil & weather inputs")
st.caption("Starting example values are provided. Replace them with actual soil-test and weather data.")
with st.form("crop_inputs"):
    c1, c2, c3 = st.columns(3)
    with c1:
        nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, 1.0)
        phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0, 1.0)
        potassium = st.number_input("Potassium (K)", 0.0, 250.0, 43.0, 1.0)
    with c2:
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, 0.5)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0, 1.0)
    with c3:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1, format="%.1f")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 200.0, 5.0)
    submitted = st.form_submit_button("🌿  Recommend Crop", use_container_width=True)

current_inputs = {
    "Nitrogen (N)": nitrogen, "Phosphorus (P)": phosphorus, "Potassium (K)": potassium,
    "Temperature (°C)": temperature, "Humidity (%)": humidity, "Soil pH": ph, "Rainfall (mm)": rainfall,
}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

if submitted:
    if model is None:
        st.error("Prediction nahi ho sakti kyunki model load nahi hua. Upar diya error fix karo.")
    else:
        try:
            input_df = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], columns=FEATURES)
            st.session_state.prediction = str(model.predict(input_df)[0])
            st.session_state.last_inputs = current_inputs.copy()
        except Exception as e:
            st.error(f"Prediction ke waqt error aaya: {e}")

st.markdown("---")
st.markdown("### Recommendation overview")
if st.session_state.prediction:
    prediction = st.session_state.prediction
    key = prediction.strip().lower()
    emoji, description, tips = CROP_INFO.get(key, ("🌱", "The model predicted this crop label. Confirm suitability with local agricultural guidance.", ["Check the soil-test report.", "Consider local climate and water availability.", "Consult an agriculture expert before planting."]))
    shown = prediction.replace("_", " ").title()
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(f'<div class="result-card"><div class="result-label">Recommended crop</div><div class="result-name">{emoji} {shown}</div><div class="muted">{description}</div></div>', unsafe_allow_html=True)
        st.caption("This is a model prediction, not a guarantee of yield or profitability.")
    with right:
        st.markdown("#### Crop guide")
        for tip in tips:
            st.markdown(f"- {tip}")
    vals = st.session_state.last_inputs or current_inputs
    st.markdown("#### Your submitted conditions")
    a, b, c, d = st.columns(4)
    a.metric("Nitrogen", f"{vals['Nitrogen (N)']:.1f}")
    b.metric("Phosphorus", f"{vals['Phosphorus (P)']:.1f}")
    c.metric("Potassium", f"{vals['Potassium (K)']:.1f}")
    d.metric("Soil pH", f"{vals['Soil pH']:.1f}")
else:
    st.markdown('<div class="panel"><h4 style="margin-top:0;color:#eaf8ee">Your crop result will appear here</h4><p class="muted">Enter the soil and weather values above, then select Recommend Crop.</p></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Input analytics", "🌾 Crop information"])
with tab1:
    st.markdown("#### Input value comparison")
    values = st.session_state.last_inputs or current_inputs
    chart_df = pd.DataFrame({"Parameter": list(values.keys()), "Value": list(values.values())}).set_index("Parameter")
    st.bar_chart(chart_df, height=300)
    st.caption("Parameters use different units and scales, so their bar heights are not directly comparable.")
    if DATA_PATH.exists():
        try:
            dataset = pd.read_csv(DATA_PATH)
            x, y = st.columns(2)
            x.metric("Dataset rows", f"{len(dataset):,}")
            y.metric("Dataset columns", len(dataset.columns))
            st.dataframe(dataset.head(5), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Dataset preview load nahi ho paya: {e}")
    else:
        st.info("Optional dataset preview ke liye data/Crop_recommendation.csv file nahi mili. Model prediction ke liye crop_model.pkl zaroori hai.")

with tab2:
    st.markdown("#### Crop reference guide")
    st.caption("General information only; local variety, season, and soil conditions matter.")
    chosen = st.selectbox("Crop select karo", sorted(CROP_INFO), format_func=lambda x: x.replace("_", " ").title())
    emoji, description, tips = CROP_INFO[chosen]
    st.markdown(f'<div class="result-card"><div style="font-size:2.5rem">{emoji}</div><h3 style="color:#f1fff5">{chosen.replace("_", " ").title()}</h3><p class="muted">{description}</p></div>', unsafe_allow_html=True)
    st.markdown("**Basic management reminders**")
    for tip in tips:
        st.markdown(f"- {tip}")

st.markdown("---")
st.caption("Smart Crop Recommendation System • Educational project • Validate recommendations locally.")
