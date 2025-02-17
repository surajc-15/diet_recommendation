
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import dill
import matplotlib.pyplot as plt

# Load the Pickle File using Dill
with open('filtered_diet.pkl', 'rb') as f:
    data = dill.load(f)

filtered_diet = data['filtered_diet']

# Function to calculate BMR and TDEE
def calculate_bmr(weight, height, age, sex):
    if sex == "male":
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
    else:
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        1: 1.2,  # sedentary
        2: 1.375,  # lightly_active
        3: 1.55,  # moderate
        4: 1.725,  # very_active
        5: 1.9   # extra_active
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)

# Function to calculate Macronutrients
def calculate_macros(calories, goal):
    if goal == "weight_loss":
        protein_ratio, fat_ratio, carb_ratio = 0.26, 0.3, 0.4
    elif goal == "weight_gain":
        protein_ratio, fat_ratio, carb_ratio = 0.25, 0.3, 0.45
    else:
        protein_ratio, fat_ratio, carb_ratio = 0.3, 0.3, 0.4
    
    protein = (calories * protein_ratio) / 4
    fat = (calories * fat_ratio) / 9
    carbs = (calories * carb_ratio) / 4
    return protein, fat, carbs


def recommend_knn(user_vector, top_n=5):
    X = filtered_diet.iloc[:, 1:10].values  # Feature vectors
    y = filtered_diet['Name'].values 
    knn = KNeighborsClassifier(n_neighbors=top_n)
    knn.fit(X, y)
    distances, indices = knn.kneighbors(user_vector)
    recommendations = filtered_diet.iloc[indices[0]]
    return recommendations

# Function to Plot Cumulative Nutritional Values
def plot_nutritional_values(recommendations, title):
    nutrition_totals = recommendations[['ProteinContent', 'FatContent', 'CarbohydrateContent']].sum()
    plt.figure(figsize=(4,4))
    plt.pie(nutrition_totals, labels=nutrition_totals.index, autopct='%1.1f%%')
    plt.title(title)
    st.pyplot(plt)

# User Inputs for Recommendation
st.title("Personalized Meal Recommendation System")
st.markdown("<div style='text-align: right; margin-bottom: 10px;'><a href='https://multilingualchatbotapplication.streamlit.app/' target='_blank'><button style='background-color:#4CAF50;color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:14px;box-shadow: 2px 2px 5px #888888;'>💬 Open Chatbot</button></a></div>", unsafe_allow_html=True)
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=60)
height = st.number_input('Height (cm)', min_value=50, max_value=250, value=165)
activity_level = st.slider('Activity Level', min_value=1, max_value=5, value=3, step=1, format='%d')
activity_names = ['Sedentary', 'Lightly Active', 'Moderate', 'Very Active', 'Extra Active']
st.markdown(f"<p style='color: green;'>Activity Level: {activity_names[activity_level-1]}</p>", unsafe_allow_html=True)
goal = st.selectbox('Goal', ['weight_loss', 'weight_gain', 'maintenance'])

target_calories = calculate_tdee(calculate_bmr(weight, height, age, sex), activity_level)
target_protein, target_fat, target_carbs = calculate_macros(target_calories, goal)

user_vector = np.array([[target_calories, target_fat, target_carbs, target_protein, 0, 0, 0, 0, 0]])

rec_knn = None  # Initialize rec_knn
if st.button('Get Recommendations'):
    rec_knn = recommend_knn(user_vector)
    st.write('### KNN Recommendations')
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('---')
if rec_knn is not None:
    rec_knn.reset_index(drop=True, inplace=True)
if rec_knn is not None:
    rec_knn['RecipeInstructions'] = rec_knn['RecipeInstructions'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
st.write('#### Recommended Meals 🌿')
if rec_knn is not None:
    cols = st.columns(len(rec_knn))
if rec_knn is not None:
    for i, col in enumerate(cols):
        with col:  # Ensure proper alignment
            col.markdown(f"<div style='border: 2px solid grey; border-radius: 10px;  background-color: #e6ffe6; margin-bottom: 10px; '>", unsafe_allow_html=True)
        with col:
            col.markdown(f"<h6 style='color: green; height: 60px; overflow: hidden; text-align: center;'>{rec_knn['Name'][i]} </h6>", unsafe_allow_html=True)
            col.markdown(f"<p style='text-align: left; min-height: 20px;'>Calories: {rec_knn['Calories'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Protein: {rec_knn['ProteinContent'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Fat: {rec_knn['FatContent'][i]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: left;'>Carbs: {rec_knn['CarbohydrateContent'][i]}</p>", unsafe_allow_html=True)
            col.expander('Recipe Instructions').write(rec_knn['RecipeInstructions'][i])
        with col:  # Ensure proper alignment
            col.markdown(f"<div style='border: 2px solid grey; border-radius: 10px;  background-color: #e6ffe6; margin-bottom: 10px; '>", unsafe_allow_html=True)
st.markdown('---')
if rec_knn is not None:
    plot_nutritional_values(rec_knn, "Nutritional Values 🌿")
