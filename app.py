def main():
    st.title('Weather Classification System')
    st.sidebar.header('Navigation')
    
    page = st.sidebar.radio(
        'Choose a Page', 
        ['Data Overview', 'Model Training', 'Weather Prediction']
    )
    
    # Load data
    data_path = 'data/weather_classification.csv'
    df = load_data(data_path)
    
    if page == 'Data Overview':
        st.header('Dataset Overview')
        st.dataframe(df.head())
        
        st.subheader('Dataset Statistics')
        st.dataframe(df.describe())
        
        st.subheader('Weather Type Distribution')
        weather_counts = df['Weather Type'].value_counts()
        st.bar_chart(weather_counts)
    
    elif page == 'Model Training':
        st.header('Model Training')
        
        # Preprocess data
        X, y, preprocessor = preprocess_data(df)
        
        if st.button('Train Model'):
            with st.spinner('Training Model...'):
                model = train_model(X, y)
                
                # Feature Importance
                st.subheader('Feature Importance')
                feature_importance(model, X.columns)
                
                st.image('feature_importance.png')
                st.image('confusion_matrix.png')
    
    elif page == 'Weather Prediction':
        st.header('Weather Prediction')
        
        # Load trained model and preprocessor
        model = joblib.load('models/weather_classifier.pkl')
        X, *, preprocessor = preprocess_data(df)
        
        # Input fields based on dataset columns
        input_data = {}
        cols = st.columns(3)
        
        with cols[0]:
            input_data['Temperature'] = st.number_input('Temperature', min_value=-50.0, max_value=50.0, step=0.1)
            input_data['Humidity'] = st.number_input('Humidity (%)', min_value=0, max_value=100, step=1)
            input_data['Wind Speed'] = st.number_input('Wind Speed', min_value=0.0, max_value=100.0, step=0.1)
        
        with cols[1]:
            input_data['Precipitation (%)'] = st.number_input('Precipitation (%)', min_value=0, max_value=100, step=1)
            input_data['Cloud Cover'] = st.selectbox('Cloud Cover', ['clear', 'partly cloudy', 'overcast'])
            input_data['Atmospheric Pressure'] = st.number_input('Atmospheric Pressure', min_value=900.0, max_value=1100.0, step=0.1)
        
        with cols[2]:
            input_data['UV Index'] = st.number_input('UV Index', min_value=0, max_value=15, step=1)
            input_data['Season'] = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
            input_data['Location'] = st.selectbox('Location', ['inland', 'mountain', 'coastal'])
        
        if st.button('Predict Weather'):
            # Prediction results with detailed visualization
            prediction, probability = predict_weather(model, preprocessor, input_data)
            
            # Weather Prediction Results Container
            st.subheader('Prediction Results')
            
            # Main Prediction Display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"## 🌦️ {prediction}")
                st.progress(probability)
                st.markdown(f"**Confidence:** {probability*100:.2f}%")
            
            # Weather Condition Visualization
            with col2:
                # Select weather icon based on prediction
                weather_icons = {
                    'Sunny': '☀️',
                    'Rainy': '🌧️', 
                    'Cloudy': '☁️',
                    'Stormy': '⛈️',
                    'Snowy': '❄️'
                }
                icon = weather_icons.get(prediction, '🌈')
                st.markdown(f"# {icon}")
            
            # Detailed Weather Insights
            st.markdown("### 📊 Detailed Insights")
            
            # Recommendation based on weather type
            weather_recommendations = {
                'Sunny': "Perfect day for outdoor activities! Don't forget sunscreen and stay hydrated.",
                'Rainy': "Carry an umbrella and wear waterproof clothing. Road conditions might be slippery.",
                'Cloudy': "Prepare for potential weather changes. Layered clothing is recommended.",
                'Stormy': "Stay indoors if possible. Avoid open areas and be prepared for sudden weather changes.",
                'Snowy': "Dress in warm layers. Check road conditions before traveling."
            }
            
            # Detailed condition breakdown
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("#### 🌡️ Temperature Analysis")
                st.write(f"**Input Temperature:** {input_data['Temperature']}°C")
                
                # Temperature interpretation
                if input_data['Temperature'] < 0:
                    st.warning("Extremely cold temperature")
                elif 0 <= input_data['Temperature'] < 10:
                    st.info("Cold temperature")
                elif 10 <= input_data['Temperature'] < 20:
                    st.success("Cool temperature")
                else:
                    st.error("Warm to hot temperature")
            
            with col_detail2:
                st.markdown("#### 💨 Wind and Humidity")
                st.write(f"**Wind Speed:** {input_data['Wind Speed']} km/h")
                st.write(f"**Humidity:** {input_data['Humidity']}%")
                
                # Wind speed interpretation
                if input_data['Wind Speed'] > 30:
                    st.warning("Strong winds detected")
                elif input_data['Wind Speed'] > 15:
                    st.info("Moderate winds")
                else:
                    st.success("Light winds")
            
            # Recommendation section
            st.markdown("### 🚨 Recommendations")
            st.info(weather_recommendations.get(prediction, "Stay prepared for various weather conditions."))

if __name__ == '__main__':
    main()
