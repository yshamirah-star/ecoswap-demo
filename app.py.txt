import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from diffusers import StableDiffusionPipeline
import torch

# --- Load trained credibility model ---
model = joblib.load("credibility_model.pkl")

# --- Load cleaned waste dataset ---
df_waste = pd.read_csv("lagos_waste_cleaned.csv")

# --- Credibility scoring function ---
def predict_credibility(features):
    score = model.predict_proba(pd.DataFrame([features]))[:,1][0]
    return round(score*100,1)

# --- Stable Diffusion pipeline for image generation ---
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def generate_waste_image(subtype):
    prompt = f"High-resolution photo of {subtype} waste material, recycling context"
    image = pipe(prompt).images[0]
    return image

# --- Streamlit UI ---
st.title("♻️ EcoSwap Marketplace Demo")

# Sidebar: select listing
listing_id = st.sidebar.selectbox("Choose a Listing ID", df_waste['listing_id'].unique())
listing = df_waste[df_waste['listing_id']==listing_id].iloc[0]

st.subheader(f"Listing {listing_id}: {listing['Hierarchical_Category']}")
st.write(f"Subtype: {listing['waste_subtype']}")
st.write(f"Quantity: {listing['quantity_kg']} kg")
st.write(f"Location: {listing['location_city']}")

# Credibility score
features = {
    "quantity_kg": listing['quantity_kg'],
    "price_per_kg(₦)": listing['price_per_kg(₦)'],
    "delivery_distance_km": listing['delivery_distance_km'],
    "estimated_cost_saving(₦)": listing['estimated_cost_saving(₦)'],
    "quality_grade_encoded": 3 if "A" in listing['quality_grade'] else 2 if "B" in listing['quality_grade'] else 1
}
score = predict_credibility(features)
st.metric("Credibility Score", f"{score}%")

# Waste image
if st.button("Generate Waste Image"):
    img = generate_waste_image(listing['waste_subtype'])
    st.image(img, caption=f"Generated image for {listing['waste_subtype']}")

# Map visualization
st.subheader("Hotspot → Recycler Flows")
lagos_map = folium.Map(location=[6.5244, 3.3792], zoom_start=12)

# Example hotspot marker
folium.CircleMarker(
    location=[6.616, 3.349],  # Ikeja Market coords
    radius=6,
    color='red',
    fill=True,
    fill_color='red',
    popup=f"Hotspot: {listing['location_city']} ({listing['Hierarchical_Category']})"
).add_to(lagos_map)

st_folium(lagos_map, width=700, height=500)