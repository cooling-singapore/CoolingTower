import os

# Pointers
WEATHER_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "inputs", "weather.xlsx")
CT_CATALOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "inputs", "catalog.xlsx")
BUILDING_DEMAND_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "inputs", "demand.xlsx")

# schemas:
BASE_CT_THRESHOLD = 0.2 #asuming the basechiller is sized to 20% of the peak load
OVERDIMENSIONING_THRESHOLD = 0.3 #asuming an extra 30% for reliability purposes.
CT_DESIGN_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs", "design.csv")
CT_AIRFLOW_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs", "airflow.csv")
CT_WATERFLOW_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs", "waterflow.csv")
CT_HWT_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs", "hwt.csv")
CT_TDRY_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs", "T_out.csv")
