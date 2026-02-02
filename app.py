"""
Streamlit Web Application for Photovoltaic System Dimensioning
Web UI for the 6-step process from the diploma thesis

This file is a Streamlit rewrite of the original Flask-like app.py.
It preserves the same 6 steps, state handling, charts, and exports.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pv_dimensioning_app import HouseholdModel, User


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Dimenzování fotovoltaiky", layout="wide")
st.title("Dimenzování fotovoltaiky – webová aplikace")
st.caption("Streamlit UI nad výpočetním modelem (6 kroků)")

# Where data files live on Streamlit Cloud / in repo
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = Path(DATA_PATH)
DATA_DIR.mkdir(exist_ok=True)

REQUIRED_FILES = [
    "DP_P1_Obsazenost_profily.xlsx",
    "DP_P2_Klimatické_podmínky_lokality.xlsx",
    "DP_P3_Spotřebiče_opak.xlsx",
    "DP_P4_Potřeba_tepla_profily_domů.xlsx",
]


# ----------------------------
# Helpers
# ----------------------------
def have_all_required_files(folder: Path) -> bool:
    return all((folder / f).exists() for f in REQUIRED_FILES)

def save_uploaded_files(folder: Path, uploads: Dict[str, Any]) -> None:
    for fname, up in uploads.items():
        out = folder / fname
        with open(out, "wb") as f:
            f.write(up.getbuffer())

@st.cache_resource
def load_model(data_path: str) -> HouseholdModel:
    # cache to avoid reloading excels on every rerun
    return HouseholdModel(data_path=data_path)

def reset_flow() -> None:
    st.session_state.step = 1
    st.session_state.users = []
    st.session_state.appliances = {}
    st.session_state.location = None
    st.session_state.house = {}
    st.session_state.ev = {}
    st.session_state.results = None

def ensure_state() -> None:
    if "step" not in st.session_state:
        reset_flow()


# ----------------------------
# Sidebar: Data handling
# ----------------------------
ensure_state()

st.sidebar.header("Data (Excel soubory)")
mode = st.sidebar.radio(
    "Odkud vzít data?",
    ["Použít složku data/ (doporučeno pro deployment)", "Nahrát Excel soubory přes web"],
    index=0,
)

if mode == "Nahrát Excel soubory přes web":
    st.sidebar.write("Nahraj 4 Excel soubory se **správnými názvy**:")
    uploads: Dict[str, Any] = {}
    for fname in REQUIRED_FILES:
        up = st.sidebar.file_uploader(fname, type=["xlsx"], key=f"upload_{fname}")
        if up is not None:
            uploads[fname] = up

    if st.sidebar.button("Uložit nahrané soubory do data/"):
        missing = [f for f in REQUIRED_FILES if f not in uploads]
        if missing:
            st.sidebar.error("Chybí:\n- " + "\n- ".join(missing))
        else:
            save_uploaded_files(DATA_DIR, uploads)
            st.sidebar.success("Soubory uloženy. Klikni na 'Reload model'.")
            load_model.clear()

else:
    if have_all_required_files(DATA_DIR):
        st.sidebar.success("Data ve složce data/ jsou kompletní ✅")
    else:
        st.sidebar.warning(
            "Ve složce data/ chybí některé soubory.\n"
            "Nahraj je do repo (data/), nebo přepni na režim nahrávání."
        )

if st.sidebar.button("Reload model (znovu načíst data)"):
    load_model.clear()

if not have_all_required_files(DATA_DIR):
    st.warning(
        "Nejsou dostupné všechny potřebné Excel soubory.\n\n"
        "➡️ Buď je dej do složky `data/`, nebo přepni vlevo na „Nahrát Excel soubory přes web“."
    )
    st.stop()

# Load model
try:
    model = load_model(str(DATA_DIR))
except Exception as e:
    st.error("Nepodařilo se načíst model. Zkontroluj Excel soubory ve složce data/.")
    st.exception(e)
    st.stop()


# ----------------------------
# Stepper UI
# ----------------------------
def goto(step: int) -> None:
    st.session_state.step = step

def back() -> None:
    st.session_state.step = max(1, int(st.session_state.step) - 1)

def next_() -> None:
    st.session_state.step = min(7, int(st.session_state.step) + 1)

steps_labels = {
    1: "1) Primární uživatel",
    2: "2) Členové domácnosti",
    3: "3) Spotřebiče",
    4: "4) Lokalita",
    5: "5) Dům",
    6: "6) Elektromobil",
    7: "Výpočet a výsledky",
}

st.sidebar.divider()
st.sidebar.subheader("Průvodce")
st.sidebar.write("Aktuální krok:")
st.sidebar.info(steps_labels.get(int(st.session_state.step), "—"))

cols_nav = st.sidebar.columns(2)
with cols_nav[0]:
    if st.button("⬅️ Zpět", use_container_width=True):
        back()
with cols_nav[1]:
    if st.button("🔄 Reset", use_container_width=True):
        reset_flow()


# ----------------------------
# STEP 1: Primary user
# ----------------------------
if int(st.session_state.step) == 1:
    st.header("Krok 1: Základní údaje – primární uživatel")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Pohlaví", ["female", "male"])
    with c2:
        age = st.number_input("Věk", min_value=0, max_value=100, value=30, step=1)
    with c3:
        status = st.selectbox(
            "Režim",
            ["student", "home_office", "classic_8_16", "morning_shift", "night_shift", "unemployed", "retired"],
        )

    if st.button("Uložit primárního uživatele →", type="primary"):
        user = User(gender=gender, age=int(age), status=status)
        st.session_state.users = [{
            "gender": gender,
            "age": int(age),
            "status": status,
            "profile_id": user.profile_id,
            "is_primary": True
        }]
        goto(2)
        st.success(f"Uživatel uložen. Profile ID: {user.profile_id}")


# ----------------------------
# STEP 2: Household members
# ----------------------------
elif int(st.session_state.step) == 2:
    st.header("Krok 2: Členové domácnosti")

    if not st.session_state.users:
        st.warning("Nejdřív vyplň primárního uživatele (Krok 1).")
        if st.button("Jít na Krok 1"):
            goto(1)
        st.stop()

    st.subheader("Aktuální členové")
    df_users = pd.DataFrame(st.session_state.users)
    st.dataframe(df_users, use_container_width=True)

    st.subheader("Přidat člena")
    c1, c2, c3 = st.columns(3)
    with c1:
        m_gender = st.selectbox("Pohlaví člena", ["female", "male"], key="m_gender")
    with c2:
        m_age = st.number_input("Věk člena", min_value=0, max_value=100, value=25, step=1, key="m_age")
    with c3:
        m_status = st.selectbox(
            "Režim člena",
            ["student", "home_office", "classic_8_16", "morning_shift", "night_shift", "unemployed", "retired"],
            key="m_status"
        )

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        if st.button("➕ Přidat člena"):
            user = User(gender=m_gender, age=int(m_age), status=m_status)
            st.session_state.users.append({
                "gender": m_gender,
                "age": int(m_age),
                "status": m_status,
                "profile_id": user.profile_id,
                "is_primary": False
            })
            st.success(f"Člen přidán. Profile ID: {user.profile_id}")

    with cbtn2:
        remove_idx = st.number_input("Index k odebrání (0 = první řádek)", min_value=0, max_value=max(0, len(st.session_state.users)-1), value=0, step=1)
        if st.button("🗑️ Odebrat člena"):
            if 0 <= int(remove_idx) < len(st.session_state.users):
                removed = st.session_state.users.pop(int(remove_idx))
                st.info(f"Odebrán: {removed}")

    st.divider()
    if st.button("Pokračovat na Krok 3 →", type="primary"):
        goto(3)


# ----------------------------
# STEP 3: Appliances
# ----------------------------
elif int(st.session_state.step) == 3:
    st.header("Krok 3: Spotřebiče")

    method = st.radio("Způsob zadání spotřeby", ["quick_estimate", "custom_selection"], index=0, horizontal=True)

    if method == "quick_estimate":
        education_level = st.selectbox("Úroveň (pro rychlý odhad)", ["basic", "high_school", "university"], index=2)
        st.session_state.appliances = {"method": "quick_estimate", "education_level": education_level}
        st.info("Rychlý odhad používá tvůj model (education_level je zatím jen parametr).")

    else:
        st.warning(
            "Custom výběr spotřebičů je v původním kódu uložen do session, "
            "ale výpočetní model ho zatím nepoužívá. Zachovávám ho jako UI funkci."
        )
        selected = st.text_area(
            "Zadej seznam spotřebičů (např. názvy oddělené čárkou)",
            value="",
            help="Placeholder – data se uloží do session, ale výpočet se řídí generate_consumption_profile()."
        )
        appliances_list = [x.strip() for x in selected.split(",") if x.strip()]
        st.session_state.appliances = {"method": "custom", "selected": appliances_list}
        st.write(f"Vybráno: {len(appliances_list)}")

    st.divider()
    if st.button("Pokračovat na Krok 4 →", type="primary"):
        goto(4)


# ----------------------------
# STEP 4: Location
# ----------------------------
elif int(st.session_state.step) == 4:
    st.header("Krok 4: Lokalita (klimatická data)")

    locations = sorted(list(model.climate_data.keys()))
    if not locations:
        st.error("V modelu nejsou dostupné lokality. Zkontroluj Excel P2.")
        st.stop()

    loc = st.selectbox("Vyber lokalitu", locations, index=0)
    st.session_state.location = loc

    st.divider()
    if st.button("Pokračovat na Krok 5 →", type="primary"):
        goto(5)


# ----------------------------
# STEP 5: House
# ----------------------------
elif int(st.session_state.step) == 5:
    st.header("Krok 5: Dům")

    method = st.radio("Způsob zadání domu", ["predefined", "custom"], index=0, horizontal=True)

    if method == "predefined":
        profile_ids = sorted(list(model.house_profiles.keys()))
        if not profile_ids:
            st.warning("House profily nejsou načteny. Můžeš použít custom.")
        profile_id = st.selectbox("Profil domu (P4)", profile_ids[:50] if profile_ids else [1])
        floors = st.selectbox("Podlaží", [1, 2], index=0)
        year = st.number_input("Rok výstavby", min_value=1900, max_value=2100, value=2000, step=1)
        floor_area = st.number_input("Podlahová plocha (m²)", min_value=20.0, max_value=600.0, value=100.0, step=5.0)

        st.session_state.house = {
            "method": "predefined",
            "profile_id": int(profile_id),
            "floors": int(floors),
            "year": int(year),
            "floor_area": float(floor_area),
        }

    else:
        floor_area = st.number_input("Podlahová plocha (m²)", min_value=20.0, max_value=600.0, value=120.0, step=5.0)
        construction_year = st.number_input("Rok výstavby", min_value=1900, max_value=2100, value=2005, step=1)
        roof_type = st.selectbox("Typ střechy", ["flat", "pitched"], index=1)
        roof_orientation = st.selectbox("Orientace střechy", ["south", "east", "west", "north", "southeast", "southwest"], index=0)
        roof_slope = st.number_input("Sklon střechy (°)", min_value=0.0, max_value=90.0, value=35.0, step=1.0)
        wall_material = st.selectbox("Materiál zdiva", ["brick", "concrete", "wood", "other"], index=0)
        wall_thickness = st.number_input("Tloušťka stěny (m)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

        st.session_state.house = {
            "method": "custom",
            "floor_area": float(floor_area),
            "construction_year": int(construction_year),
            "roof_type": roof_type,
            "roof_orientation": roof_orientation,
            "roof_slope": float(roof_slope),
            "wall_material": wall_material,
            "wall_thickness": float(wall_thickness),
        }

    st.info(
        "Pozn.: V aktuální verzi výpočetního modelu se parametry domu přímo nepromítají do elektrické spotřeby "
        "(počítá se hlavně obsazenost + stochastické spotřebiče). UI i uložení ale zachovávám."
    )

    st.divider()
    if st.button("Pokračovat na Krok 6 →", type="primary"):
        goto(6)


# ----------------------------
# STEP 6: EV
# ----------------------------
elif int(st.session_state.step) == 6:
    st.header("Krok 6: Elektromobil (volitelně)")

    has_ev = st.checkbox("Domácnost má elektromobil", value=False)
    if has_ev:
        c1, c2, c3 = st.columns(3)
        with c1:
            battery_capacity = st.number_input("Kapacita baterie (kWh)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)
        with c2:
            annual_km = st.number_input("Roční nájezd (km)", min_value=1000.0, max_value=100000.0, value=15000.0, step=500.0)
        with c3:
            count = st.number_input("Počet EV", min_value=1, max_value=5, value=1, step=1)

        st.session_state.ev = {
            "has_ev": True,
            "battery_capacity": float(battery_capacity),
            "annual_km": float(annual_km),
            "count": int(count),
        }
    else:
        st.session_state.ev = {"has_ev": False}

    st.divider()
    if st.button("Spočítat výsledky →", type="primary"):
        goto(7)


# ----------------------------
# STEP 7: Calculate & results
# ----------------------------
elif int(st.session_state.step) == 7:
    st.header("Výpočet a výsledky")

    if not st.session_state.users:
        st.warning("Chybí uživatelé (Krok 1–2).")
        if st.button("Jít na Krok 1"):
            goto(1)
        st.stop()

    # Parameters that were "hard-coded" in the original calculation
    st.subheader("Nastavení výpočtu")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_power = st.number_input("Min výkon FVE (kWp)", min_value=0.5, max_value=30.0, value=1.0, step=0.5)
    with c2:
        max_power = st.number_input("Max výkon FVE (kWp)", min_value=1.0, max_value=30.0, value=15.0, step=0.5)
    with c3:
        step = st.number_input("Krok (kWp)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    seed = st.number_input("Náhodné semínko (stabilní výsledky)", min_value=0, max_value=10_000_000, value=42, step=1)

    if st.button("▶️ Spustit výpočet", type="primary"):
        np.random.seed(int(seed))

        temp_model = HouseholdModel(data_path=str(DATA_DIR))

        # Add users
        for u in st.session_state.users:
            temp_model.add_user(gender=u["gender"], age=u["age"], status=u["status"])

        household_profile = temp_model.generate_household_profile()
        occupancy_rate = float(np.mean(household_profile) * 100)

        appliance_cfg = st.session_state.appliances or {}
        education_level = appliance_cfg.get("education_level", "university")

        consumption = temp_model.generate_consumption_profile(
            household_profile=household_profile,
            education_level=education_level,
        )
        base_consumption = float(np.sum(consumption))
st.write("DEBUG: education_level použité ve výpočtu:", education_level)
st.write("DEBUG: roční spotřeba (kWh):", float(np.sum(consumption)))

        # EV
        ev_cfg = st.session_state.ev or {}
        if ev_cfg.get("has_ev"):
            for _ in range(int(ev_cfg.get("count", 1))):
                consumption = temp_model.add_ev_consumption(
                    consumption=consumption,
                    battery_capacity=float(ev_cfg.get("battery_capacity", 0)),
                    annual_km=float(ev_cfg.get("annual_km", 0)),
                )

        total_consumption = float(np.sum(consumption))
        ev_consumption = float(total_consumption - base_consumption)

        # Location
        loc_name = st.session_state.location
        if loc_name and loc_name in temp_model.climate_data:
            location = temp_model.climate_data[loc_name]
        else:
            location = list(temp_model.climate_data.values())[0]
            loc_name = location.name

        # Optimize
        if max_power < min_power:
            st.error("Max výkon musí být >= min výkon.")
            st.stop()

        with st.spinner("Optimalizuji velikost FVE…"):
            optimal_pv, optimal_battery, results = temp_model.optimize_pv_size(
                consumption=consumption,
                location=location,
                min_power=float(min_power),
                max_power=float(max_power),
                step=float(step),
            )

        if not results:
            st.error("Nepodařilo se najít optimální řešení. Zkus jiné rozsahy výkonů.")
            st.stop()

        eb = results["energy_balance"]
        ec = results["economics"]

        response_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "household": {
                "member_count": len(st.session_state.users),
                "occupancy_rate": round(occupancy_rate, 1),
                "location": loc_name,
            },
            "consumption": {
                "base": round(base_consumption, 0),
                "ev": round(ev_consumption, 0),
                "total": round(total_consumption, 0),
            },
            "optimal_system": {
                "pv_power": optimal_pv,
                "battery_capacity": optimal_battery,
            },
            "energy_balance": {
                "self_consumption": round(float(eb["self_consumption_kwh"]), 0),
                "grid_export": round(float(eb["grid_export_kwh"]), 0),
                "grid_import": round(float(eb["grid_import_kwh"]), 0),
                "self_consumption_rate": round(float(eb["self_consumption_rate"]), 1),
                "autarky_rate": round(float(eb["autarky_rate"]), 1),
                "battery_cycles": round(float(eb.get("battery_cycles", 0)), 1),
            },
            "economics": {
                "pv_cost": round(float(ec["pv_cost"]), 0),
                "battery_cost": round(float(ec["battery_cost"]), 0),
                "total_cost": round(float(ec["total_cost"]), 0),
                "total_cost_with_subsidy": round(float(ec["total_cost_with_subsidy"]), 0),
                "annual_savings": round(float(ec["annual_savings"]), 0),
                "payback_period": round(float(ec["payback_period"]), 1),
                "npv": round(float(ec["npv"]), 0),
                "irr": round(float(ec["irr"]), 1),
                "recommended": bool(ec["recommended"]),
            },
            "inputs": {
                "users": st.session_state.users,
                "appliances": st.session_state.appliances,
                "location": st.session_state.location,
                "house": st.session_state.house,
                "ev": st.session_state.ev,
                "seed": int(seed),
                "opt_range": {"min_power": float(min_power), "max_power": float(max_power), "step": float(step)},
            },
        }

        st.session_state.results = response_data
        st.success("Výpočet dokončen ✅")

    # Display results if exist
    if st.session_state.results:
        r = st.session_state.results

        st.subheader("Souhrn")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Doporučený výkon FVE", f"{r['optimal_system']['pv_power']:.1f} kWp")
        m2.metric("Doporučená baterie", f"{r['optimal_system']['battery_capacity']:.1f} kWh")
        m3.metric("Roční spotřeba", f"{r['consumption']['total']:,.0f} kWh".replace(",", " "))
        m4.metric("Obsazenost", f"{r['household']['occupancy_rate']:.1f} %")

        st.subheader("Energetická bilance")
        eb = r["energy_balance"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Vlastní spotřeba", f"{eb['self_consumption']:,.0f} kWh".replace(",", " "))
        c2.metric("Export", f"{eb['grid_export']:,.0f} kWh".replace(",", " "))
        c3.metric("Import", f"{eb['grid_import']:,.0f} kWh".replace(",", " "))
        c4, c5 = st.columns(2)
        c4.metric("Míra samospotřeby", f"{eb['self_consumption_rate']:.1f} %")
        c5.metric("Míra soběstačnosti", f"{eb['autarky_rate']:.1f_toggle:=}" if False else f"{eb['autarky_rate']:.1f} %")

        st.subheader("Ekonomika")
        ec = r["economics"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Investice (po dotaci)", f"{ec['total_cost_with_subsidy']:,.0f} Kč".replace(",", " "))
        e2.metric("Roční úspora", f"{ec['annual_savings']:,.0f} Kč".replace(",", " "))
        e3.metric("Návratnost", f"{ec['payback_period']:.1f} let")
        e4.metric("IRR", f"{ec['irr']:.1f} %")

        st.write("NPV (25 let):", f"**{ec['npv']:,.0f} Kč**".replace(",", " "))
        st.write("Doporučení:", "✅ Doporučeno" if ec["recommended"] else "⚠️ Nedoporučeno")

        # Charts
        st.subheader("Grafy")

        # Pie: energy balance
        fig1 = plt.figure(figsize=(7, 5))
        labels = ["Self-consumption", "Grid export", "Grid import"]
        sizes = [eb["self_consumption"], eb["grid_export"], eb["grid_import"]]
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Annual energy balance")
        plt.axis("equal")
        st.pyplot(fig1, clear_figure=True)

        # Bar: economics overview
        fig2 = plt.figure(figsize=(7, 5))
        categories = ["Investment", "Annual savings (x10)", "NPV (÷1000)"]
        values = [
            ec["total_cost_with_subsidy"],
            ec["annual_savings"] * 10,
            ec["npv"] / 1000,
        ]
        plt.bar(categories, values)
        plt.title("Economic overview")
        plt.ylabel("CZK")
        plt.grid(axis="y", alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

        # Consumption breakdown pie (base vs EV)
        fig3 = plt.figure(figsize=(7, 5))
        labels = ["Base", "EV"]
        sizes = [r["consumption"]["base"], r["consumption"]["ev"]]
        if r["consumption"]["ev"] > 0:
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.axis("equal")
        else:
            plt.text(0.5, 0.5, f"Total: {r['consumption']['total']:,.0f} kWh/yr\n(No EV)",
                     ha="center", va="center", fontsize=14)
            plt.axis("off")
        plt.title("Consumption breakdown")
        st.pyplot(fig3, clear_figure=True)

        # Exports
        st.subheader("Export")
        json_bytes = json.dumps(r, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Stáhnout výsledky (JSON)",
            data=json_bytes,
            file_name="pv_results.json",
            mime="application/json",
        )

        # Simple CSV export
        flat = {
            "member_count": r["household"]["member_count"],
            "occupancy_rate_pct": r["household"]["occupancy_rate"],
            "location": r["household"]["location"],
            "consumption_base_kwh": r["consumption"]["base"],
            "consumption_ev_kwh": r["consumption"]["ev"],
            "consumption_total_kwh": r["consumption"]["total"],
            "optimal_pv_kwp": r["optimal_system"]["pv_power"],
            "optimal_battery_kwh": r["optimal_system"]["battery_capacity"],
            "self_consumption_kwh": r["energy_balance"]["self_consumption"],
            "grid_export_kwh": r["energy_balance"]["grid_export"],
            "grid_import_kwh": r["energy_balance"]["grid_import"],
            "self_consumption_rate_pct": r["energy_balance"]["self_consumption_rate"],
            "autarky_rate_pct": r["energy_balance"]["autarky_rate"],
            "investment_czk": r["economics"]["total_cost_with_subsidy"],
            "annual_savings_czk": r["economics"]["annual_savings"],
            "payback_years": r["economics"]["payback_period"],
            "npv_czk": r["economics"]["npv"],
            "irr_pct": r["economics"]["irr"],
            "recommended": r["economics"]["recommended"],
        }
        df = pd.DataFrame([flat])
        st.download_button(
            "⬇️ Stáhnout souhrn (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="pv_results_summary.csv",
            mime="text/csv",
        )

    else:
        st.info("Klikni na „Spustit výpočet“ pro získání výsledků.")
