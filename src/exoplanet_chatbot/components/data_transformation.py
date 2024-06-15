import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from exoplanet_chatbot.logging import logger
from exoplanet_chatbot.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config:DataTransformationConfig): # It will take the configuration from DataIngestionConfig defined earlier , which will in turn use Configuration Manager to take data from config.yaml
        self.config = config
    
    def feature_selection(self,data):

        columns_to_include = ['Planet Name','Host Name','Number of Stars','Number of Planets','Number of Moons','Circumbinary Flag','Discovery Method', 'Discovery Year',
       'Discovery Publication Date','Discovery Facilty','Discovery Telescope','Planet Radius [Earth Radius]',
       'Planet Mass [Earth Mass]','Planet Density [g/cm**3]','Equilibrium Temperature [K]','Orbit Semi-Major Axis [au]',
       'Radial Velocity Amplitude [m/s]','Stellar Effective Temperature [K]','Stellar Radius [Solar Radius]','Stellar Mass [Solar mass]']
        
        return data[columns_to_include]

    def feature_engineering(self,data):

        data.rename(columns={'Circumbinary Flag' : 'Binary System'},inplace=True)
        data['Binary System'] = data['Binary System'].map({0 : 'Binary System', 1 : 'Not Binary System'})
        return data
    
    def null_value_handling(self,data):

        data.fillna("Not found Yet!!",inplace=True)
        return data

    def context_generator(self,row):

        context = (
        f"The exoplanet {row['Planet Name']} orbits the host star {row['Host Name']}. "
        f"The system containing the exoplanet {row['Planet Name']} also contains {row['Number of Stars']} stars, {row['Number of Planets']} planets, and {row['Number of Moons']} moons. "
        f"The exoplanet lies in a {row['Binary System']}. "
        f"The planet was discovered using the {row['Discovery Method']} method in {row['Discovery Year']}. "
        f"The discovery was published on {row['Discovery Publication Date']} and facilitated by {row['Discovery Facilty']} using the {row['Discovery Telescope']}. "
        f"The planet has a radius of {row['Planet Radius [Earth Radius]']} Earth radii, a mass of {row['Planet Mass [Earth Mass]']} Earth masses, "
        f"and a density of {row['Planet Density [g/cm**3]']} g/cm³. "
        f"The equilibrium temperature is {row['Equilibrium Temperature [K]']} K. "
        f"The semi-major axis of its orbit is {row['Orbit Semi-Major Axis [au]']} AU. "
        f"The radial velocity amplitude is {row['Radial Velocity Amplitude [m/s]']} m/s. "
        f"The host star has an effective temperature of {row['Stellar Effective Temperature [K]']} K, "
        f"a radius of {row['Stellar Radius [Solar Radius]']} solar radii, and a mass of {row['Stellar Mass [Solar mass]']} solar masses."
    )
        return context
    
    def instruction_pair_generator(self,data):

        instruction_context_response_pairs = []

        for index, row in data.iterrows():
            planet_name = row['Planet Name']
            context = row['Context']
            features = {
                'Number of Stars': row['Number of Stars'],
                'Number of Planets': row['Number of Planets'],
                'Number of Moons': row['Number of Moons'],
                'Binary System': row['Binary System'],
                'Discovery Method': row['Discovery Method'],
                'Discovery Year': row['Discovery Year'],
                'Discovery Publication Date': row['Discovery Publication Date'],
                'Discovery Facility': row['Discovery Facilty'],
                'Discovery Telescope': row['Discovery Telescope'],
                'Planet Radius': row['Planet Radius [Earth Radius]'],
                'Planet Mass': row['Planet Mass [Earth Mass]'],
                'Planet Density': row['Planet Density [g/cm**3]'],
                'Equilibrium Temperature': row['Equilibrium Temperature [K]'],
                'Orbit Semi-Major Axis': row['Orbit Semi-Major Axis [au]'],
                'Radial Velocity Amplitude': row['Radial Velocity Amplitude [m/s]'],
                'Stellar Effective Temperature': row['Stellar Effective Temperature [K]'],
                'Stellar Radius': row['Stellar Radius [Solar Radius]'],
                'Stellar Mass': row['Stellar Mass [Solar mass]']
            }

            for feature, value in features.items():
                instruction = f"What is the {feature.lower().replace('_', ' ')} of {planet_name}?"
                response = f"The {feature.lower().replace('_', ' ')} of {planet_name} is {value}."
                instruction_context_response_pairs.append({
                    "instruction": instruction,
                    "input": context,
                    "output": response
                })
        
        return pd.DataFrame(instruction_context_response_pairs)

    def transform(self):

        # Reading the data
        data = pd.read_csv(self.config.data_path)

        # Feature selection
        feature_selected_data = self.feature_selection(data)

        # Null value handling
        null_value_handled_data = self.null_value_handling(feature_selected_data)

        # Feature Engineering
        feature_engineered_data = self.feature_engineering(null_value_handled_data)

        # Context generation
        feature_engineered_data['Context'] = feature_engineered_data.apply(self.context_generator, axis=1)

        # Instruction pair generation
        instruction_context_response_pairs = self.instruction_pair_generator(feature_engineered_data)

        # Saving the data
        instruction_context_response_pairs.to_csv(self.config.data_path_transformed, index=False)
