![image](https://github.com/user-attachments/assets/3b530250-a873-4225-8ca4-cb21ac245d4b)


# Rental Apartments Data Dashboard

This project provides a data visualization and analysis tool for rental apartment data. It consists of two main components:

## 1. Data Cleaning and Analysis (`clean-data.ipynb`)

- **Outlier Detection**: Identifies and removes outliers in numerical columns like `price` and `square_feet` using Z-scores.
- **Statistical Analysis**: Generates insights and statistics from the raw dataset, such as correlations and trends.
- **Output**: Produces a cleaned dataset (`rents_cleaned.csv`) for use in the interactive dashboard.

## 2. Interactive Dashboard (`main.py`)

- **Filters**: Users can filter the data by state, city, bedrooms, price range, and square footage.
- **Visualizations**:
  - **Choropleth Map**: Displays average rental prices by state.
  - **Bar Charts**: Illustrates trends in average rental prices over time.
  - **Heatmap**: Highlights correlations between features like price, bedrooms, and square footage.
  - **Scatter Plot**: Shows the relationship between price and square footage.
  - **Folium Map**: An interactive map displaying average prices by city.
  - **Top Locations**: Horizontal bar chart ranking cities or states by average rental price.
- **Dynamic Metrics**: Displays the number of rental listings matching the current filter criteria.

---

## How to Run

1. **Install Required Dependencies**  
   Run the following command to install all necessary libraries:
   ```
   pip install -r requirements.txt
   ```
2. **Generate Cleaned Dataset**
    Open the Jupyter notebook and execute all cells:
    ```
    jupyter notebook clean-data.ipynb
    ```
    This will create the rents_cleaned.csv file.
3.  **Start the Dashboard**
    Launch the dashboard by running:
    ```
    python main.py
    ```
4.  **Access the Dashboard**
    Open your browser and navigate to:
    ```
    http://127.0.0.1:8070
    ```
