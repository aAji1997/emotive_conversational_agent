#!/usr/bin/env python3
"""
Setup script for the memory database.
This script sets up the Supabase database with the necessary tables and functions.
"""
# Import required modules
import importlib.util
from pathlib import Path

# Define the path to the setup_database.py file
setup_database_path = Path(__file__).resolve().parent / 'supabase' / 'setup_database.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location('setup_database_module', setup_database_path)
setup_database_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_database_module)

# Get the setup_database function
setup_database = setup_database_module.setup_database

if __name__ == "__main__":
    print("Setting up memory database...")
    success = setup_database()
    if success:
        print("Memory database setup complete!")
    else:
        print("Memory database setup failed!")
