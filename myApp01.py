import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid


# cred = credentials.Certificate('"C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house-prices-ed310-firebase-adminsdk-z4vwb-d6da5dfcd6.json"')
# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'your-bucket-name.appspot.com'
# })

# # Get Firestore database instance
# db = firestore.client()

# # Get Firebase Storage bucket instance
# bucket = storage.bucket()




# Load the trained models
with open('Linear_regressor_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)
with open('DecisionTreeRegressor_model.pkl', 'rb') as file1:
    decision_tree_model = pickle.load(file1)
with open('RandomForestRegressor_model.pkl', 'rb') as file2:
    random_forest_model = pickle.load(file2)
# with open('CatBoostRegressor_model.pkl', 'rb') as file3:
#     catboost_model = pickle.load(file3)
with open('lgbm_model.pkl', 'rb') as file4:
        lgbm_model = pickle.load(file4)
        
# Load label encoders
with open('district_encoder.pkl', 'rb') as file5:
    district_encoder = pickle.load(file5)
with open('micro_district_encoder.pkl', 'rb') as file6:
    micro_district_encoder = pickle.load(file6)
with open('building_type_encoder.pkl', 'rb') as file7:
    building_type_encoder = pickle.load(file7)
with open('condition_encoder.pkl', 'rb') as file8:
    condition_encoder = pickle.load(file8)

# Streamlit interface
st.title("House Price Prediction")

# Sidebar - Page selection and model selection
st.sidebar.header('Choose Page')
page_options = ['Analyze Prices', 'Predict Prices', 'Find an Apartment']
page = st.sidebar.selectbox("Which page do you want to visit?", page_options)


# Load the data
house_data = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house_kg_10K_ads.csv')

# Define independent and dependent variables
independent_variables = ["square", "rooms", "floors", "floor", "date_year"]
dependent_variable = "price"

# Define the categorical variables
categorical_variables = ['district', 'micro_district', 'building_type', 'condition']

# Create and fit the label encoders for each categorical variable
label_encoders = {var: LabelEncoder().fit(house_data[var]) for var in categorical_variables}

# Transform and add the encoded columns to house_data
for var, encoder in label_encoders.items():
    encoded_column = var + '_encoded'
    house_data[encoded_column] = encoder.transform(house_data[var])
    independent_variables.append(encoded_column)  # Add the encoded column to independent variables

# Sidebar - Dropdowns for categorical inputs and encoding
district_options = ['Октябрьский район', 'Ленинский район', 'Первомайский район', 'Свердловский район']
microdistrict_options = ['Магистраль', 'Академия Наук', 'ЖД вокзал', 'Unknown', 'Аламедин-1 м-н',
       '6 м-н', 'Кок-Жар ж/м', 'Асанбай м-н', 'Джал-23 м-н (Нижний Джал)',
       'Военторг', 'АЮ Grand', 'Восток-5 м-н', 'Молодая Гвардия',
       'Верхний Джал м-н', 'КНУ', '4 м-н', 'Политех', 'Джал 15 м-н',
       'Ипподром', 'Площадь Победы', '11 м-н', 'Мед. академия', 'Ак Кеме',
       'Моссовет', 'Горького - Панфилова', 'Московская - Белинка',
       'Старый аэропорт', 'АУЦА', 'Дворец спорта', '12 м-н', 'Гоин',
       'Московская - Уметалиева', 'Парк Ататюрк', 'Жилгородок Ницца',
       'Карла Маркса', 'ЦУМ', 'Сквер Тоголок Молдо', 'Бишкек-Парк',
       'Душанбинка', 'Восточный автовокзал', 'Центральная мечеть',
       'Юбилейка', 'Космос', '8 м-н', 'Кара-Жыгач ж/м',
       'Джальская больница', 'Средний Джал м-н', 'Золотой квадрат',
       'Ден Сяопина - Фучика', 'Нижний Токольдош', '5 м-н', 'Матросова',
       'Парк Панфилова/Спартак', '7 м-н', 'Карпинка', 'Кудайберген',
       'Джал-29 м-н', 'Улан м-н', 'Пишпек ж/м', 'ТЭЦ', 'БГУ', 'VEFA',
       'Щербакова ж/м', 'Ак Эмир рынок', 'Госрегистр', 'Кок-Жар м-н',
       'Церковь', 'Чуй - Алматинка', '3 м-н', 'Азия Молл',
       'Цирк/Дворец бракосочетания', 'Шлагбаум', 'Филармония',
       'Джал-30 м-н', 'Нижний Джал м-н', 'Джал 30', 'КГУСТА', '10 м-н',
       'Западный автовокзал', 'Таатан', 'Гагарина', 'Учкун м-н',
       'Тунгуч м-н', 'Городок энергетиков', '9 м-н',
       'Советская - Скрябина', 'Ак-Орго ж/м', 'Арча-Бешик ж/м', 'Баха',
       'Городок строителей', 'Вечерка', 'Кара Дарыя', 'Юг-2 м-н',
       '1000 мелочей', 'Ген прокуратура', 'Таш Рабат',
       'Аламединский рынок', 'Жилгородок Совмина ж/м', 'Улан-2 м-н',
       'Дордой Плаза', '110 квартал ж/м', 'Кызыл-Аскер ж/м', 'Достук',
       'Мадина', 'Ак-Босого ж/м', 'Алматинка - Магистраль', 'Комфорт',
       'Ошский рынок', 'Токольдош ж/м', 'Киргизия 1 ж/м',
       'Алтын-Ордо ж/м', 'Рухий Мурас ж/м', 'Достук м-н', 'Учкун ж/м',
       'Орозбекова - Жибек-Жолу', 'Рабочий Городок', 'Ала-Арча ж/м',
       'с. Орто-Сай', 'с. Чон-Арык', 'Дордой ж/м', 'Ак-Ордо ж/м',
       'Колмо ж/м', 'Физкультурный', 'Эне-Сай ж/м', 'Киргизия-2 м-н',
       'Ынтымак ж/м', 'Салам-Алик ж/м', 'Старый толчок',
       'Балбан-Таймаш ж/м', 'Детский мир', '69-га', 'Учкун-2 ж/м',
       'Касым ж/м', 'Ортосайский рынок', 'Красный Строитель 2 ж/м']  # Assume this contains the full list of microdistricts
building_type_options = ['кирпичный', 'монолитный', 'панельный']
condition_options = ['под самоотделку (ПСО)', 'хорошее', 'евроремонт', 'среднее', 'не достроено', 'требует ремонта', 'черновая отделка', 'свободная планировка']  # Assume this contains the full list of conditions



# Streamlit page setup
if page == 'Analyze Prices':
    st.write("This page will provide analysis of house prices.")

    # Header for price analysis
    st.subheader("House Price Analysis")

    # Select box for choosing an independent variable
    independent_var_choice = st.selectbox("Select a variable to analyze with house prices", independent_variables)

    # Creating a scatter plot for the chosen independent variable vs. house prices
    scatter_plot = px.scatter(house_data, x=independent_var_choice, y=dependent_variable,
                              title=f"House Prices vs {independent_var_choice}")
    st.plotly_chart(scatter_plot)

    # Header for detailed analysis
    st.subheader("Detailed Analysis of House Prices")

    # Option for choosing a type of analysis
    analysis_type = st.selectbox("Choose the type of analysis", 
                                 ('Pairplot', 'Correlation Matrix', 'Price Distribution', 'Price Trends', 'Price by District'))

    if analysis_type == 'Pairplot':
        # Code for displaying a pairplot
        # Selecting a subset of the data for the pairplot to avoid performance issues
        # We limit to a maximum of 500 samples to prevent slowdowns
        pairplot_data = house_data[independent_variables + [dependent_variable]].sample(min(500, len(house_data)), random_state=1)
        st.write("Displaying pairplot for a random sample of up to 500 listings for performance reasons.")
        pairplot_fig = sns.pairplot(pairplot_data)
        st.pyplot(pairplot_fig)
    elif analysis_type == 'Correlation Matrix':
        # Code for displaying a correlation matrix
        corr_matrix = house_data[independent_variables + [dependent_variable]].corr()
        matrix_plot = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(matrix_plot)
    elif analysis_type == 'Price Distribution':
        # Code for displaying a distribution plot
        dist_plot = px.histogram(house_data, x=dependent_variable, nbins=50)
        st.plotly_chart(dist_plot)
    elif analysis_type == 'Price Trends':
        # Code for displaying a trend line plot
        trend_plot = px.line(house_data, x='date_year', y=dependent_variable)
        st.plotly_chart(trend_plot)
    elif analysis_type == 'Price by District':
        # Code for displaying a pie chart of house prices by district
        # Replace 'district' with the actual column name for districts in your dataset
        price_by_district = house_data.groupby('district')[dependent_variable].sum().reset_index()
        pie_chart = px.pie(price_by_district, values=dependent_variable, names='district', title='House Prices by District')
        st.plotly_chart(pie_chart)




    
    
    
    
    
# Main area - Display user inputs and perform prediction
elif page == 'Predict Prices':
    st.sidebar.header('Model Selection and User Inputs')
    model_options = ['Linear Regression', 'Decision Tree', 'Random Forest', 'LGBM', 'CATBOOST']
    selected_model = st.sidebar.selectbox("Choose a model for prediction", model_options)

    # Sidebar - Sliders for user input
    square = st.sidebar.slider("Enter Square Meters", min_value=0, max_value=1000, step=1)
    rooms = st.sidebar.slider("Enter Number of Rooms", min_value=0, max_value=10, step=1)
    floors = st.sidebar.slider("Enter Number of Floors in Building", min_value=0, max_value=100, step=1)
    floor = st.sidebar.slider("Enter Floor Number", min_value=0, max_value=100, step=1)
    date_year = st.sidebar.slider("Enter Year of Construction", min_value=1900, max_value=2024, step=1)

    # Sidebar - Dropdowns for categorical inputs
    microdistrict = st.sidebar.selectbox("Select Microdistrict", microdistrict_options)
    district = st.sidebar.selectbox("Select District", district_options)
    condition = st.sidebar.selectbox("Select Condition", condition_options)
    building_type = st.sidebar.selectbox("Select Building Type", building_type_options)

    # Encoding categorical inputs
    district_encoded = district_encoder.transform([district])[0]
    micro_district_encoded = micro_district_encoder.transform([microdistrict])[0]
    building_type_encoded = building_type_encoder.transform([building_type])[0]
    condition_encoded = condition_encoder.transform([condition])[0]

    # Prepare input data array
    input_data = np.array([[square, rooms, floors, floor, date_year, 
                            district_encoded, micro_district_encoded, 
                            building_type_encoded, condition_encoded]])

    # Prediction and model summary based on the selected model
    if selected_model == 'Linear Regression':
        prediction = linear_model.predict(input_data)
        model_summary_data = {
            "Model": "Linear Regression",
            "MSE": 489020200.79406863,
            "R Squared": 0.7507187181330213
        }
    elif selected_model == 'Decision Tree':
        prediction = decision_tree_model.predict(input_data)
        model_summary_data = {
            "Model": "Decision Tree Regressor",
            "MSE": 574342995.740365,
            "R Squared": 0.7354839965380982
        }
    elif selected_model == 'Random Forest':
        prediction = random_forest_model.predict(input_data)
        model_summary_data = {
            "Model": "Random Forest Regressor",
            "MSE": 192164806.4368714,
            "R Squared": 0.9020427188886709
        }
    elif selected_model == 'LGBM':
        prediction = lgbm_model.predict(input_data)
        model_summary_data = {
            "Model": "LGBM",
            "MSE": 370814781.9969399,
            "R Squared": 0.8377896010239424
        }
    elif selected_model == 'CATBOOST':
         prediction = lgbm_model.predict(input_data)
         model_summary_data = {
               "Model": "LGBM",
               "MSE": 360614704.47670025,
               "R Squared": 0.8422515554132322
           }
    
   

    # Display the user inputs as a table
    st.header('User Input Values')
    user_inputs_df = pd.DataFrame({
        'Input Feature': ['Square Meters', 'Number of Rooms', 'Number of Floors in Building', 'Floor Number', 'Year of Construction', 'District', 'Microdistrict', 'Building Type', 'Condition'],
        'Value': [square, rooms, floors, floor, date_year, district, microdistrict, building_type, condition]
    })
    st.table(user_inputs_df)

    # Display the prediction and model summary
    st.header('Predictions')
    if prediction is not None:
        st.success(f"The predicted house price is: {prediction[0]:,.2f}")

    st.header('Model Summary')
    if model_summary_data is not None:
        model_summary_df = pd.DataFrame([model_summary_data])
        st.table(model_summary_df)

elif page == 'Find an Apartment':
    st.title('Find an Apartment')
    
    
    
    # Upload button for user to upload a picture of their apartment
    uploaded_file = st.file_uploader("Upload a picture of your apartment", type=["jpg", "png", "jpeg"])

    # Define variables outside of the submit condition to ensure they are in scope
    square = rooms = floors = floor = date_year = district = microdistrict = building_type = condition = None

    if uploaded_file is not None:
        # Layout for displaying uploaded image and details side by side
        col1, col2 = st.columns(2)
        with col1:
            # User inputs various details about the apartment
            square = st.number_input("Square Footage", min_value=0, key="square1")
            # ... (the rest of your inputs)

        with col1:
            # User inputs various details about the apartment
            square = st.number_input("Square Footage", min_value=0, key="square2")
            rooms = st.number_input("Number of Rooms", min_value=1, step=1, key="rooms1")
            floors = st.number_input("Total Floors in Building", min_value=1, step=1, key="floors1")
            floor = st.number_input("Floor Number", min_value=1, step=1, key="floors")
            date_year = st.number_input("Year of Construction", min_value=1900, max_value=2024, step=1, key="date_year1")

        with col2:
            # User inputs various details about the apartment
            district_options = ['Октябрьский район', 'Ленинский район', 'Первомайский район', 'Свердловский район']
            district = st.selectbox("District", options=district_options, key="district")

            microdistrict_options = ['Option1', 'Option2', 'Option3']  # Replace with actual options
            microdistrict = st.selectbox("Microdistrict", options=microdistrict_options, key="microdistrict")

            building_type_options = ['кирпичный', 'монолитный', 'панельный']
            building_type = st.selectbox("Building Type", options=building_type_options, key="building_type")

            condition_options = ['под самоотделку (ПСО)', 'хорошее', 'евроремонт', 'среднее', 'не достроено', 'требует ремонта', 'черновая отделка', 'свободная планировка']
            condition = st.selectbox("Condition", options=condition_options, key="condition")
            
        
        # Import UUID to generate a unique image filename
       
        filename = f"{uuid.uuid4()}.{uploaded_file.name.split('.')[-1]}"

       # Convert the uploaded file to bytes
        file_bytes = uploaded_file.getvalue()

       # Upload the image to Firebase Storage
        blob = bucket.blob(filename)
        blob.upload_from_string(file_bytes, content_type=uploaded_file.type)

       # Make the blob publicly viewable
        blob.make_public()

       # Get the URL of the uploaded file
        image_url = blob.public_url

       # Submit button
        submit_button = st.button("Submit")

        if submit_button:

         if submit_button:
             
            # Add data to Firestore
             doc_ref = db.collection('apartments').add({
                 'square': square,
                 'rooms': rooms,
                 'floors': floors,
                 'floor': floor,
                 'year': date_year,
                 'district': district,
                 'microdistrict': microdistrict,
                 'building_type': building_type,
                 'condition': condition,
                # Assume the image URL will be set later once the file is uploaded
                 'image_url': None
            })

            # Set the image URL once the image is uploaded
            # Note: You may need to adjust this if you want to make the image publicly accessible
            #image_url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
            #db.collection('apartments').document(doc_ref.id).update({'image_url': image_url})