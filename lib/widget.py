import ipywidgets as widgets
from IPython.display import display
import os
import pandas as pd
import numpy as np
from scipy import io
from lib import train, result_analysis



def training_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Define the list of algorithm options
    algorithm_options = ['UserKNN', 'MF']
    
    # Create a radio button widget for datasets
    rad_button_dataset = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create a radio button widget for algorithms (Initially hidden)
    rad_button_algorithm = widgets.RadioButtons(
        options=algorithm_options,
        description='Choose algorithm:',
        disabled=False,
        value=None
    )
    
    # Create an OK button for dataset selection
    ok_button_dataset = widgets.Button(
        description='OK',
        disabled=True
    )
    
    # Create an OK button for algorithm selection (Initially hidden)
    ok_button_algorithm = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # VBox container to control the display of the second radio button and OK button
    algorithm_selection_box = widgets.VBox([rad_button_algorithm, ok_button_algorithm])
    algorithm_selection_box.layout.display = 'none'  # Initially hidden
    
    # Variables to store the selected dataset and algorithm
    selected_dataset = None
    selected_algorithm = None
    
    # Function to handle the dataset radio button change
    def on_radio_dataset_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            print(selected_dataset)
            ok_button_dataset.disabled = False  # Enable OK button when a dataset is selected
    
    # Function to handle the algorithm radio button change
    def on_radio_algorithm_change(change):
        global selected_algorithm
        if change['type'] == 'change' and change['name'] == 'value':
            selected_algorithm = change['new']
            ok_button_algorithm.disabled = False  # Enable OK button when an algorithm is selected
    
    # Function to handle the dataset OK button click
    def on_ok_button_dataset_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Show the algorithm selection radio button and enable the algorithm OK button
                algorithm_selection_box.layout.display = 'block'
    
    # Function to handle the algorithm OK button click
    def on_ok_button_algorithm_clicked(b):
        global selected_algorithm
        with output_radb:
            output_radb.clear_output()
            if selected_algorithm:
                print(f'Algorithm chosen: {selected_algorithm}')
                print(f'Starting process for dataset: {selected_dataset} with algorithm: {selected_algorithm}')
                # Trigger the training process here (e.g., train with chosen algorithm and dataset)
                train.train(selected_dataset, selected_algorithm)
    
    # Attach the functions to the respective widgets
    rad_button_dataset.observe(on_radio_dataset_change)
    rad_button_algorithm.observe(on_radio_algorithm_change)
    ok_button_dataset.on_click(on_ok_button_dataset_clicked)
    ok_button_algorithm.on_click(on_ok_button_algorithm_clicked)

    # Display the dataset radio button and its OK button
    display(rad_button_dataset, ok_button_dataset, output_radb)
    
    # Display the second radio button and OK button for algorithm selection (hidden initially)
    display(algorithm_selection_box)

def analysis_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Define the list of algorithm options
    algorithm_options = ['UserKNN', 'MF']
    
    # Create a radio button widget for datasets
    rad_button_dataset = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create a radio button widget for algorithms (Initially hidden)
    rad_button_algorithm = widgets.RadioButtons(
        options=algorithm_options,
        description='Choose algorithm:',
        disabled=False,
        value=None
    )
    
    # Create an OK button for dataset selection
    ok_button_dataset = widgets.Button(
        description='OK',
        disabled=True
    )
    
    # Create an OK button for algorithm selection (Initially hidden)
    ok_button_algorithm = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # VBox container to control the display of the second radio button and OK button
    algorithm_selection_box = widgets.VBox([rad_button_algorithm, ok_button_algorithm])
    algorithm_selection_box.layout.display = 'none'  # Initially hidden
    
    # Variables to store the selected dataset and algorithm
    selected_dataset = None
    selected_algorithm = None
    
    # Function to handle the dataset radio button change
    def on_radio_dataset_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            ok_button_dataset.disabled = False  # Enable OK button when a dataset is selected
    
    # Function to handle the algorithm radio button change
    def on_radio_algorithm_change(change):
        global selected_algorithm
        if change['type'] == 'change' and change['name'] == 'value':
            selected_algorithm = change['new']
            ok_button_algorithm.disabled = False  # Enable OK button when an algorithm is selected
    
    # Function to handle the dataset OK button click
    def on_ok_button_dataset_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Show the algorithm selection radio button and enable the algorithm OK button
                algorithm_selection_box.layout.display = 'block'
    
    # Function to handle the algorithm OK button click
    def on_ok_button_algorithm_clicked(b):
        global selected_algorithm
        with output_radb:
            output_radb.clear_output()
            if selected_algorithm:
                print(f'Algorithm chosen: {selected_algorithm}')
                print(f'Starting process for dataset: {selected_dataset}, with algorithm: {selected_algorithm}')
                # Trigger the training process here (e.g., train with chosen algorithm and dataset)
                result_analysis.analyse(selected_dataset, selected_algorithm)
    
    # Attach the functions to the respective widgets
    rad_button_dataset.observe(on_radio_dataset_change)
    rad_button_algorithm.observe(on_radio_algorithm_change)
    ok_button_dataset.on_click(on_ok_button_dataset_clicked)
    ok_button_algorithm.on_click(on_ok_button_algorithm_clicked)

    # Display the dataset radio button and its OK button
    display(rad_button_dataset, ok_button_dataset, output_radb)
    
    # Display the second radio button and OK button for algorithm selection (hidden initially)
    display(algorithm_selection_box)