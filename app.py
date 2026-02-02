# app.py
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pv_dimensioning_app import HouseholdModel


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="PV Dimensioning", layout="wide")
st.title("Dimenzování fotovoltaiky – webová aplikace")
st.caption("Streamlit UI nad výpočetním modelem v pv_dimensioning_app.py")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_FILES = [
    "DP_P1_Obsazenost_profily.xlsx",
    "DP_P2_Klimatické_podmínky_lokality.xlsx",
    "DP_P3_Spotřebiče_opak.xlsx",
    "DP_P4_Potřeba_tepla_profily_domů.xlsx",
]


# ---------------------------
# Helpers
# ---------------------------
def _have_all_required_files(folder: Path) -> bool:
    return all((folder / f).exists() for f in REQUIRED_FILES)


def _save_uploaded_files_to_data_dir(uploaded: dict[str, st.runtime.uploaded_file_manager.UploadedFile], folder: Path):
    """
    Save uploaded Streamlit files into folder with exact required filenames.
    uploaded: dict filename -> UploadedFile
    """
    for fname in REQUIRED_FILES:
        uf = uploaded.get(fname)
        if uf is None:
            continue
        out_path = folder / fname
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())


@st.cache_resource
def load_model(data_path: str) -> HouseholdModel:
    """
    Cache model instance. If data files change, user can hit "Reload model".
    """
    return HouseholdModel(data_path=data_path)


def _force_reload_model():
    load_model.clear()


def _safe_get_locations(model: HouseholdModel) -> list[str]:
    keys = list(model.climate_data.keys()) if hasattr(model, "climate_data") and model.climate_data else []
    # Fallback if empty for any reason
    if not keys:
        keys = ["Prague"]
    return keys


# ---------------------------
# Data source UI
# ---------------------------
st.sidebar.header("Data (Excel soubory)")

mode = st.sidebar.radio(
    "Odkud vzít data?",
    ["Použít složku data/ (doporučeno pro deployment)", "Nahrát Excel soubory přes web"],
    index=0,
)

if mode == "Nahrát Excel soubory přes web":
    st.sidebar.write("Nahraj 4 Excel soubory se **správnými názvy**:")
    uploaded_files = {}
    for fname in REQUIRED_FILES:
        uf = st.sidebar.file_uploader(fname, type=["xlsx"], key=f"up_{fname}")
        if uf is not None:
            uploaded_files[fname] = uf

    if st.sidebar.button("Uložit nahrané soubory do data/"):
        missing = [f for f in REQUIRED_FILES if f not in uploaded_files]
        if missing:
            st.sidebar.error("Chybí tyto soubory:\n- " + "\n- ".join(missing))
        else:
            _save_uploaded_files_to_data_dir(uploaded_files, DATA_DIR)
            st.sidebar.success("Soubory uloženy do data/.")
            _force_reload_model()

else:
    st.sidebar.write("Aplikace očekává soubory ve složce `data/`.")
    if _have_all_required_files(DATA_DIR):
        st.sidebar.success("Data ve složce data/ jsou kompletní ✅")
    else:
        st.sidebar.warning(
            "Ve složce data/ chybí některé soubory.\n"
            "Buď je tam nahraj (přes GitHub), nebo přepni na režim nahrávání."
        )

if st.sidebar.button("Reload model (znovu načíst data)"):
    _force_reload_model()


# ---------------------------
# Load model
# ---------------------------
if not _have_all_required_files(DATA_DIR):
    st.warning(
        "Nejsou dostupné všechny potřebné Excel soubory.\n\n"
        "➡️ Buď je dej do složky `data/`, nebo přepni v levém panelu na „Nahrát Excel soubory přes web“."
    )
    st.stop()

try:
    model = load_model(str(DATA_DIR))
except Exception as e:
    st.error("Nepodařilo se načíst model. Zkontroluj data soubory ve složce data/.")
    st.exception(e)
    st.stop()


# ---------------------------
# Inputs
# ---------------------------
st.header("1) Domácnost")

colA, colB = st.columns([1, 1])

with colA:
    people_count = st.number_input("Počet členů domácnosti", min_value=1, max_value=10, value=3, step=1)

with colB:
    location_name = st.selectbox("Lokalita (klimatická data)", _safe_get_locations(model))

st.subheader("Členové domácnosti (profily obsazenosti)")

# Collect people
people = []
for i in range(int(people_count)):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        gender = st.selectbox(f"Pohlaví #{i+1}", ["female", "male"], key=f"gender_{i}")
    with c2:
        age = st.number_input(f"Věk #{i+1}", min_value=0, max_value=100, value=30, step=1, key=f"age_{i}")
    with c3:
        status = st.selectbox(
            f"Režim #{i+1}",
            ["student", "home_office", "classic_8_16", "morning_shift", "night_shift", "unemployed", "retired"],
            key=f"status_{i}",
        )
    people.append((gender, int(age), status))

st.header("2) Elektromobil (volitelné)")
ev_col1, ev_col2 = st.columns(2)
with ev_col1:
    ev_battery = st.number_input("Kapacita baterie EV (kWh)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
with ev_col2:
    ev_km = st.number_input("Roční nájezd (km)", min_value=0.0, max_value=100000.0, value=0.0, step=500.0)

st.header("3) Optimalizace FVE")
o1, o2, o3 = st.columns(3)
with o1:
    min_power = st.number_input("Min. výkon (kWp)", min_value=0.5, max_value=30.0, value=1.0, step=0.5)
with o2:
    max_power = st.number_input("Max. výkon (kWp)", min_value=1.0, max_value=30.0, value=15.0, step=0.5)
with o3:
    step = st.number_input("Krok (kWp)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

run = st.button("Spočítat", type="primary")


# ---------------------------
# Run calculation
# ---------------------------
if run:
    if max_power < min_power:
        st.error("Max. výkon musí být větší nebo rovný min. výkonu.")
        st.stop()

    # Reset users in model
    model.users = []

    # Add users
    for g, a, s in people:
        model.add_user(gender=g, age=a, status=s)

    # Occupancy
    household_profile = model.generate_household_profile()

    # Consumption (your current method is stochastic)
    base_consumption = model.generate_consumption_profile(household_profile=household_profile)

    # Add EV if any
    total_consumption = model.add_ev_consumption(
        consumption=base_consumption,
        battery_capacity=float(ev_battery),
        annual_km=float(ev_km),
    )

    # Location
    try:
        location = model.climate_data.get(location_name) or list(model.climate_data.values())[0]
    except Exception:
        st.error("Nepodařilo se vybrat lokalitu z klimatických dat.")
        st.stop()

    # Optimize
    with st.spinner("Počítám optimální velikost FVE…"):
        optimal_pv, optimal_batt, results = model.optimize_pv_size(
            consumption=total_consumption,
            location=location,
            min_power=float(min_power),
            max_power=float(max_power),
            step=float(step),
        )

    st.success(f"Doporučený výkon FVE: **{optimal_pv:.1f} kWp** | Doporučená baterie: **{optimal_batt:.1f} kWh**")

    # Show summaries
    annual_consumption = float(np.sum(total_consumption))
    st.metric("Roční spotřeba (odhad)", f"{annual_consumption:,.0f} kWh".replace(",", " "))

    if not results:
        st.warning("Optimalizace nenašla řešení, které splnilo podmínky modelu (např. IRR >= 5%). Zkus jiné rozsahy výkonů.")
        st.stop()

    eb = results["energy_balance"]
    ec = results["economics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Vlastní spotřeba z FVE", f"{eb['self_consumption_kwh']:,.0f} kWh".replace(",", " "))
    c2.metric("Export do sítě", f"{eb['grid_export_kwh']:,.0f} kWh".replace(",", " "))
    c3.metric("Dovoz ze sítě", f"{eb['grid_import_kwh']:,.0f} kWh".replace(",", " "))

    c4, c5 = st.columns(2)
    c4.metric("Míra samospotřeby", f"{eb['self_consumption_rate']:.1f} %")
    c5.metric("Míra soběstačnosti", f"{eb['autarky_rate']:.1f} %")

    st.subheader("Ekonomika")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Investice (po dotaci)", f"{ec['total_cost_with_subsidy']:,.0f} Kč".replace(",", " "))
    e2.metric("Roční úspora", f"{ec['annual_savings']:,.0f} Kč".replace(",", " "))
    e3.metric("Návratnost", f"{ec['payback_period']:.1f} let")
    e4.metric("IRR", f"{ec['irr']:.1f} %")

    st.write(f"NPV (25 let): **{ec['npv']:,.0f} Kč**".replace(",", " "))
    st.write("Doporučení:", "✅ **RECOMMENDED**" if ec.get("recommended") else "⚠️ **NOT RECOMMENDED**")

    # ---------------------------
    # Simple charts: daily average profiles
    # ---------------------------
    st.subheader("Grafy (orientačně)")

    # Convert hourly arrays into average day (24h) profiles
    cons = np.array(total_consumption)
    # Use the selected optimal pv to compute production for plot
    prod = model.calculate_pv_production(installed_power=float(optimal_pv), location=location)

    # Average by hour of day
    cons_day = np.zeros(24)
    prod_day = np.zeros(24)
    for h in range(24):
        cons_day[h] = float(np.mean(cons[h::24]))
        prod_day[h] = float(np.mean(prod[h::24]))

    df_day = pd.DataFrame(
        {"Spotřeba (kWh/h)": cons_day, "Výroba FVE (kWh/h)": prod_day},
        index=[f"{h:02d}:00" for h in range(24)],
    )

    st.line_chart(df_day)

    # Monthly sums
    # naive month slicing: assumes non-leap year 365 days
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    month_cons = []
    month_prod = []
    start = 0
    for dim in days_in_month:
        end = start + dim * 24
        month_cons.append(float(np.sum(cons[start:end])))
        month_prod.append(float(np.sum(prod[start:end])))
        start = end

    df_month = pd.DataFrame(
        {"Spotřeba (kWh)": month_cons, "Výroba FVE (kWh)": month_prod},
        index=month_labels,
    )

    st.bar_chart(df_month)

    st.info(
        "Pozn.: spotřeba v modelu je částečně náhodná (stochastická), takže při každém spuštění se může mírně lišit."
    )
