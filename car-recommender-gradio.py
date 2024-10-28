import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np

class CarRecommender:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name="car_database",
            embedding_function=self.embedding_fn
        )
        
    def load_data(self, csv_file):
        """Load and process car data from CSV file"""
        self.df = pd.read_csv(csv_file)
        self.df = self.df.fillna('-')
        # Convert price to numeric, removing any currency symbols and commas
        self.df['Price'] = pd.to_numeric(self.df['Price'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
        
    def create_car_description(self, row):
        """Create a detailed description from car attributes"""
        return f"{row['Year']} {row['Brand']} {row['Model']} - {row['BodyType']} with {row['Engine']} engine, "\
               f"{row['Transmission']} transmission, {row['DriveType']} drive type. "\
               f"Fuel type: {row['FuelType']}, Fuel consumption: {row['FuelConsumption']}. "\
               f"Features {row['Doors']}, {row['Seats']}. Color: {row['ColourExtInt']}. "\
               f"Condition: {row['UsedOrNew']}, Mileage: {row['Kilometres']} km."
    
    def populate_database(self):
        """Populate the ChromaDB collection with car data"""
        ids = [str(i) for i in range(len(self.df))]
        descriptions = [self.create_car_description(row) for _, row in self.df.iterrows()]
        
        metadatas = []
        for _, row in self.df.iterrows():
            metadata = {
                'brand': row['Brand'],
                'model': row['Model'],
                'year': str(row['Year']),
                'price': str(row['Price']),
                'body_type': row['BodyType'],
                'transmission': row['Transmission'],
                'fuel_type': row['FuelType'],
                'drive_type': row['DriveType'],
                'kilometers': str(row['Kilometres']),
                'condition': row['UsedOrNew']
            }
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            documents=descriptions,
            metadatas=metadatas
        )
    
    def get_unique_values(self):
        """Get unique values for dropdown filters"""
        return {
            'brands': sorted(self.df['Brand'].unique()),
            'body_types': sorted(self.df['BodyType'].unique()),
            'transmissions': sorted(self.df['Transmission'].unique()),
            'fuel_types': sorted(self.df['FuelType'].unique()),
            'conditions': sorted(self.df['UsedOrNew'].unique()),
            'min_price': int(self.df['Price'].min()),
            'max_price': int(self.df['Price'].max()),
            'min_year': int(self.df['Year'].min()),
            'max_year': int(self.df['Year'].max())
        }

    def recommend_cars(self, 
                      description, 
                      brand=None,
                      body_type=None,
                      transmission=None,
                      fuel_type=None,
                      condition=None,
                      min_price=None,
                      max_price=None,
                      min_year=None,
                      max_year=None,
                      n_results=5):
        """
        Recommend cars based on description and filters
        """
        # Build where clause for filters
        where = {}
        if brand and brand != "Any":
            where["brand"] = brand
        if body_type and body_type != "Any":
            where["body_type"] = body_type
        if transmission and transmission != "Any":
            where["transmission"] = transmission
        if fuel_type and fuel_type != "Any":
            where["fuel_type"] = fuel_type
        if condition and condition != "Any":
            where["condition"] = condition
            
        # Price and year filters are handled post-query due to string/numeric conversion needs
        
        # Query the collection
        results = self.collection.query(
            query_texts=[description],
            n_results=n_results * 2,  # Query more results to allow for post-filtering
            where=where
        )
        
        # Format and filter recommendations
        recommendations = []
        for i in range(len(results['ids'][0])):
            car_index = int(results['ids'][0][i])
            car_data = self.df.iloc[car_index]
            
            # Apply numeric filters
            if min_price and car_data['Price'] < min_price:
                continue
            if max_price and car_data['Price'] > max_price:
                continue
            if min_year and car_data['Year'] < min_year:
                continue
            if max_year and car_data['Year'] > max_year:
                continue
            
            car_info = {
                'brand': car_data['Brand'],
                'model': car_data['Model'],
                'year': car_data['Year'],
                'price': f"${car_data['Price']:,.2f}",
                'body_type': car_data['BodyType'],
                'transmission': car_data['Transmission'],
                'engine': car_data['Engine'],
                'drive_type': car_data['DriveType'],
                'fuel_type': car_data['FuelType'],
                'fuel_consumption': car_data['FuelConsumption'],
                'kilometers': f"{car_data['Kilometres']:,} km",
                'color': car_data['ColourExtInt'],
                'location': car_data['Location'],
                'condition': car_data['UsedOrNew'],
                'similarity_score': results['distances'][0][i] if 'distances' in results else None
            }
            recommendations.append(car_info)
            
            if len(recommendations) >= n_results:
                break
                
        return recommendations

def format_recommendations(recommendations):
    """Format recommendations for Gradio output"""
    if not recommendations:
        return "No cars found matching your criteria."
        
    output = ""
    for idx, car in enumerate(recommendations, 1):
        output += f"Recommendation #{idx}:\n"
        output += f"Car: {car['year']} {car['brand']} {car['model']}\n"
        output += f"Price: {car['price']}\n"
        output += f"Body Type: {car['body_type']}\n"
        output += f"Engine: {car['engine']}\n"
        output += f"Transmission: {car['transmission']}\n"
        output += f"Drive Type: {car['drive_type']}\n"
        output += f"Fuel Type: {car['fuel_type']}\n"
        output += f"Fuel Consumption: {car['fuel_consumption']}\n"
        output += f"Kilometers: {car['kilometers']}\n"
        output += f"Color: {car['color']}\n"
        output += f"Location: {car['location']}\n"
        output += f"Condition: {car['condition']}\n"
        if car['similarity_score'] is not None:
            output += f"Match Score: {1 - car['similarity_score']:.2f}\n"
        output += "\n" + "-"*50 + "\n\n"
    
    return output

def create_gradio_interface():
    # Initialize recommender and load data
    recommender = CarRecommender()
    recommender.load_data('your_car_data.csv')  # Replace with your CSV file path
    recommender.populate_database()
    
    # Get unique values for filters
    unique_values = recommender.get_unique_values()
    
    # Define the interface
    with gr.Blocks(title="Car Recommendation System") as interface:
        gr.Markdown("# Car Recommendation System")
        gr.Markdown("Describe your ideal car and use the filters to narrow down your search.")
        
        with gr.Row():
            with gr.Column():
                description = gr.Textbox(
                    label="Describe your ideal car",
                    placeholder="Example: I need a fuel-efficient family SUV with good safety features",
                    lines=3
                )
                
                with gr.Row():
                    brand = gr.Dropdown(
                        choices=["Any"] + list(unique_values['brands']),
                        label="Brand",
                        value="Any"
                    )
                    body_type = gr.Dropdown(
                        choices=["Any"] + list(unique_values['body_types']),
                        label="Body Type",
                        value="Any"
                    )
                
                with gr.Row():
                    transmission = gr.Dropdown(
                        choices=["Any"] + list(unique_values['transmissions']),
                        label="Transmission",
                        value="Any"
                    )
                    fuel_type = gr.Dropdown(
                        choices=["Any"] + list(unique_values['fuel_types']),
                        label="Fuel Type",
                        value="Any"
                    )
                
                with gr.Row():
                    min_price = gr.Slider(
                        minimum=unique_values['min_price'],
                        maximum=unique_values['max_price'],
                        label="Minimum Price ($)",
                        value=unique_values['min_price']
                    )
                    max_price = gr.Slider(
                        minimum=unique_values['min_price'],
                        maximum=unique_values['max_price'],
                        label="Maximum Price ($)",
                        value=unique_values['max_price']
                    )
                
                with gr.Row():
                    min_year = gr.Slider(
                        minimum=unique_values['min_year'],
                        maximum=unique_values['max_year'],
                        label="Minimum Year",
                        value=unique_values['min_year']
                    )
                    max_year = gr.Slider(
                        minimum=unique_values['min_year'],
                        maximum=unique_values['max_year'],
                        label="Maximum Year",
                        value=unique_values['max_year']
                    )
                
                condition = gr.Dropdown(
                    choices=["Any"] + list(unique_values['conditions']),
                    label="Condition",
                    value="Any"
                )
                
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    label="Number of recommendations",
                    value=5
                )
                
                submit_btn = gr.Button("Get Recommendations", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="Recommendations",
                    lines=25,
                    max_lines=25
                )
        
        submit_btn.click(
            fn=lambda *args: format_recommendations(
                recommender.recommend_cars(*args)
            ),
            inputs=[
                description,
                brand,
                body_type,
                transmission,
                fuel_type,
                condition,
                min_price,
                max_price,
                min_year,
                max_year,
                num_results
            ],
            outputs=output
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(share=True)
