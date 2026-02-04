"""
Photovoltaic System Dimensioning Application
Based on predictive electricity consumption models

This application helps households independently assess the energy efficiency
and economic effectiveness of photovoltaic system installation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


@dataclass
class User:
    """Represents a household member with their profile"""
    gender: str  # 'male' or 'female'
    age: int
    status: str  # work regime: 'student', 'home_office', 'classic_8_16', 'morning_shift', 'night_shift', 'unemployed', 'retired'
    profile_id: int = None
    
    def __post_init__(self):
        """Assign profile ID based on user characteristics"""
        self.profile_id = self._determine_profile()
    
    def _determine_profile(self) -> int:
        """
        Determine profile number based on gender, age, and status
        Returns profile ID from 1-28 based on the decision tree
        """
        base_offset = 0 if self.gender == 'female' else 14  # Profiles 1-14 female, 15-28 male
        
        if self.age < 16:
            return base_offset + 1  # Young child
        elif 16 <= self.age <= 27:
            if self.status == 'student':
                return base_offset + 2
            elif self.status in ['home_office', 'classic_8_16', 'morning_shift', 'night_shift']:
                return base_offset + 3
            else:
                return base_offset + 4
        elif 28 <= self.age <= 65:
            if self.status == 'unemployed':
                return base_offset + 13
            elif self.status == 'home_office':
                return base_offset + 6
            elif self.status == 'classic_8_16':
                return base_offset + 8
            elif self.status == 'morning_shift':
                return base_offset + 9
            elif self.status == 'night_shift':
                return base_offset + 10
            else:
                return base_offset + 11
        else:  # > 65
            return base_offset + 27  # Senior


@dataclass
class Appliance:
    """Represents an electrical appliance"""
    name: str
    power_old: float  # Power consumption when old (W)
    power_new: float  # Power consumption when new (W)
    duration_old: float  # Duration of use when old (h)
    duration_new: float  # Duration of use when new (h)
    uses_per_day_old: float  # Number of uses per day (old)
    uses_per_day_new: float  # Number of uses per day (new)
    energy_old: float  # Energy per use (kWh) - old
    energy_new: float  # Energy per use (kWh) - new
    time_zone: str  # 'morning', 'evening', 'all_day'
    gender_association: str  # 'male', 'female', 'both'


@dataclass
class House:
    """Represents house technical parameters"""
    floor_area: float  # m²
    roof_type: str  # 'flat', 'pitched'
    roof_orientation: str  # 'south', 'east', 'west', 'north'
    roof_slope: float  # degrees
    construction_year: int
    wall_construction: str
    wall_material: str
    wall_thickness: float  # m
    thermal_demand: np.ndarray = None  # Hourly heat demand for the year


@dataclass
class Location:
    """Climate data for location"""
    name: str
    latitude: float
    longitude: float
    temperature: np.ndarray  # 8760 hourly values
    global_radiation: np.ndarray  # 8760 hourly values


class HouseholdModel:
    """Main model for household energy consumption and PV system dimensioning"""
    
    def __init__(self, data_path: str = '/mnt/user-data/uploads'):
        self.data_path = Path(data_path)
        self.users: List[User] = []
        self.appliances: List[Appliance] = []
        self.house: Optional[House] = None
        self.location: Optional[Location] = None
        self.ev_battery_capacity: float = 0  # kWh
        self.ev_annual_km: float = 0
        
        # Data will be loaded
        self.occupation_profiles: Dict[int, np.ndarray] = {}
        self.appliance_database: pd.DataFrame = None
        self.climate_data: Dict[str, Location] = {}
        self.house_profiles: Dict[int, House] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data from Excel files"""
        print("Loading data files...")
        
        # Load occupation profiles (P1)
        self._load_occupation_profiles()
        
        # Load appliance database (P3)
        self._load_appliances()
        
        # Load climate data (P2)
        self._load_climate_data()
        
        # Load house thermal demand profiles (P4)
        self._load_house_profiles()
        
        print("Data loading complete.")
    
    def _load_occupation_profiles(self):
        """Load 28 occupation profiles from P1"""
        try:
            df = pd.read_excel(
                self.data_path / 'DP_P1_Obsazenost_profily.xlsx',
                sheet_name=0,
                header=None
            )
            
            # Each profile is a column of 8760 values (0/1)
            # Skip header rows and read profiles
            for col_idx in range(1, min(29, df.shape[1])):  # Profiles 1-28
                profile_data = df.iloc[2:8762, col_idx].values
                # Convert to binary
                profile_data = np.array([1 if x == 1 else 0 for x in profile_data])
                if len(profile_data) == 8760:
                    self.occupation_profiles[col_idx] = profile_data
            
            print(f"Loaded {len(self.occupation_profiles)} occupation profiles")
            
        except Exception as e:
            print(f"Error loading occupation profiles: {e}")
            # Create dummy profiles for testing
            for i in range(1, 29):
                self.occupation_profiles[i] = np.random.randint(0, 2, 8760)
    
    def _load_appliances(self):
        """Load appliance database from P3"""
        try:
            df = pd.read_excel(
                self.data_path / 'DP_P3_Spotřebiče_opak.xlsx',
                sheet_name=0
            )
            self.appliance_database = df
            print(f"Loaded appliance database with {len(df)} entries")
            
        except Exception as e:
            print(f"Error loading appliances: {e}")
            self.appliance_database = pd.DataFrame()
    
    def _load_climate_data(self):
        """Load climate data for 48 locations from P2"""
        try:
            df = pd.read_excel(
                self.data_path / 'DP_P2_Klimatické_podmínky_lokality.xlsx',
                sheet_name=0,
                header=None
            )
            
            # Row 0 has location names, row 1 has "Te" and "GHI" labels
            # Data starts from row 3 (index 3)
            location_names = df.iloc[0, :].dropna()
            
            # Parse each location (every 2 columns: Te and GHI)
            col_idx = 2  # Start from column 2 (first data column)
            locations_loaded = 0
            
            while col_idx < df.shape[1]:  # Load all locations from P2
                try:
                    location_name = df.iloc[0, col_idx]
                    if pd.isna(location_name):
                        col_idx += 2
                        continue
                    
                    # Get temperature and radiation data
                    temp_data = pd.to_numeric(df.iloc[3:8763, col_idx], errors='coerce').fillna(10).values
                    rad_data = pd.to_numeric(df.iloc[3:8763, col_idx + 1], errors='coerce').fillna(0).values
                    
                    # Ensure we have 8760 values
                    if len(temp_data) >= 8760 and len(rad_data) >= 8760:
                        # Radiation seems to be in different units - scale up if needed
                        # Typical global radiation should be 0-1000 W/m²
                        if np.max(rad_data) < 100:
                            rad_data = rad_data * 30  # Scale up
                        
                        location = Location(
                            name=str(location_name),
                            latitude=50.0,
                            longitude=14.0,
                            temperature=temp_data[:8760],
                            global_radiation=np.maximum(rad_data[:8760], 0)  # Ensure non-negative
                        )
                        self.climate_data[str(location_name)] = location
                        locations_loaded += 1
                        
                except Exception as e:
                    pass
                
                col_idx += 2  # Each location has 2 columns (Te, GHI)
            
            print(f"Loaded climate data for {len(self.climate_data)} locations")
            
        except Exception as e:
            print(f"Error loading climate data: {e}")
            # Create dummy climate data with realistic values
            # Prague typical: 1000 kWh/m²/year = ~114 W/m² average
            self.climate_data['Prague'] = Location(
                name='Prague',
                latitude=50.0,
                longitude=14.5,
                temperature=10 + 15 * np.sin(np.arange(8760) * 2 * np.pi / 8760 - np.pi/2),  # Seasonal temp
                global_radiation=np.maximum(500 * np.sin(np.arange(8760) * 2 * np.pi / 8760) * 
                                          np.sin(np.arange(8760) * np.pi / 24), 0)  # Daily and seasonal pattern
            )
    
    def _load_house_profiles(self):
        """Load house thermal demand profiles from P4"""
        try:
            df = pd.read_excel(
                self.data_path / 'DP_P4_Potřeba_tepla_profily_domů.xlsx',
                sheet_name=0,
                header=None
            )
            
            # Data starts from row 5 (index 5), first 4 rows are headers
            # Each column represents a different house profile
            for col_idx in range(1, min(51, df.shape[1])):  # 50 house profiles
                # Extract thermal data from row 5 onwards
                thermal_data = pd.to_numeric(df.iloc[5:8765, col_idx], errors='coerce').fillna(0).values
                if len(thermal_data) >= 8760:
                    house = House(
                        floor_area=100,  # Will be updated
                        roof_type='pitched',
                        roof_orientation='south',
                        roof_slope=35,
                        construction_year=2000,
                        wall_construction='brick',
                        wall_material='brick',
                        wall_thickness=0.4,
                        thermal_demand=(thermal_data[:8760].astype(float) / 1000.0)  # P4 is Wh/h -> convert to kWh/h
                    )
                    self.house_profiles[col_idx] = house
            
            print(f"Loaded {len(self.house_profiles)} house profiles")
            
        except Exception as e:
            print(f"Error loading house profiles: {e}")
    

    # -------------------------------------------------
    # House profile assignment (P4) based on building params
    # -------------------------------------------------
    def _bin_floor_area(self, area: float) -> int:
        """5 bins for floor area -> 1..5"""
        area = float(area)
        if area < 70:
            return 1
        if area < 100:
            return 2
        if area < 130:
            return 3
        if area < 170:
            return 4
        return 5

    def _bin_year(self, year: int) -> int:
        """5 bins for (construction/renovation) year -> 1..5"""
        y = int(year)
        if y < 1970:
            return 1
        if y < 1990:
            return 2
        if y < 2005:
            return 3
        if y < 2015:
            return 4
        return 5

    def _bin_floors(self, floors: int) -> int:
        """2 bins for floors -> 1..2"""
        return 1 if int(floors) <= 1 else 2

    def assign_house_profile_id(self, floor_area: float, floors: int, year: int) -> int:
        """
        Deterministic mapping (like user profiles):
        5 (area) × 5 (year) × 2 (floors) = 50 profiles.

        Returns:
            profile_id in range 1..50
        """
        a = self._bin_floor_area(floor_area)   # 1..5
        y = self._bin_year(year)               # 1..5
        f = self._bin_floors(floors)           # 1..2
        return int((a - 1) * 10 + (y - 1) * 2 + (f - 1) + 1)

    def get_house_heat_profile(self, floor_area: float, floors: int, year: int) -> tuple[int, np.ndarray]:
        """
        Get assigned P4 heat demand profile (8760) and scale by floor area.

        Returns:
            (assigned_profile_id, heat_demand_kwh_th_per_h[8760])
        """
        profile_id = self.assign_house_profile_id(floor_area=floor_area, floors=floors, year=year)

        # P4 loader stores profiles as House objects with thermal_demand.
        if profile_id not in self.house_profiles:
            raise ValueError(
                f"House profile {profile_id} not found in loaded P4 data. "
                f"Loaded IDs: {sorted(self.house_profiles.keys())[:10]}..."
            )

        base_house = self.house_profiles[profile_id]
        thermal = np.array(base_house.thermal_demand, dtype=float)

        if thermal.shape[0] != 8760:
            raise ValueError(f"House profile {profile_id} must have 8760 values, got {thermal.shape[0]}")

        # Scale: assume P4 profiles represent ~100 m² baseline
        scale = float(floor_area) / 100.0
        thermal_scaled = thermal * scale
        return int(profile_id), thermal_scaled

    def add_heating_to_consumption(
        self,
        consumption_kwh: np.ndarray,
        heat_demand_kwh_th: np.ndarray,
        heating_system: str,
        location: Optional[Location] = None,
        hp_cop_nominal: float = 3.0,
    ) -> np.ndarray:
        """
        Convert heat demand to electricity and add it to consumption.
        heating_system:
          - 'none'
          - 'direct_electric'  (COP = 1)
          - 'heat_pump'        (COP depends on outdoor temperature if available, else hp_cop_nominal)
        """
        out = np.array(consumption_kwh, dtype=float).copy()

        system = (heating_system or "none").lower().strip()
        if system == "none":
            return out

        heat = np.array(heat_demand_kwh_th, dtype=float)
        if heat.shape[0] != 8760:
            raise ValueError(f"heat_demand_kwh_th must have 8760 values, got {heat.shape[0]}")

        if system == "direct_electric":
            out += heat
            return out

        if system == "heat_pump":
            # Simple COP model based on temperature, fallback to constant COP.
            cop = None
            if location is not None and hasattr(location, "temperature"):
                temp = np.array(location.temperature, dtype=float)
                if temp.shape[0] == 8760:
                    # COP approx: 0°C -> 2.0, 10°C -> 2.6, 20°C -> 3.2
                    cop = np.clip(2.0 + 0.06 * temp, 1.5, 4.0)

            if cop is None:
                cop = np.full(8760, float(hp_cop_nominal))

            out += heat / cop
            return out

        raise ValueError(f"Unknown heating_system: {heating_system}")

    def add_user(self, gender: str, age: int, status: str):
        """Add a household member"""
        user = User(gender=gender, age=age, status=status)
        self.users.append(user)
        print(f"Added user: {gender}, {age}, {status} -> Profile {user.profile_id}")
        return user
    
    def generate_household_profile(self) -> np.ndarray:
        """
        Generate aggregated household occupation profile
        Returns 8760 array where 1 = someone is home
        """
        if not self.users:
            return np.zeros(8760)
        
        # Start with all zeros
        household_profile = np.zeros(8760, dtype=int)
        
        # For each hour, if ANY user is home (1), household is occupied
        for user in self.users:
            if user.profile_id in self.occupation_profiles:
                user_profile = self.occupation_profiles[user.profile_id]
                household_profile = np.logical_or(household_profile, user_profile).astype(int)
        
        occupancy_rate = np.mean(household_profile) * 100
        print(f"Household occupancy rate: {occupancy_rate:.1f}%")
        
        return household_profile
    def generate_consumption_profile(self, 
                                     household_profile: np.ndarray,
                                     education_level: str = 'university') -> np.ndarray:
        """
        Generate hourly electricity consumption profile based on:
        - Household occupation
        - Stochastic appliance usage

        Returns 8760 array of power consumption in kWh per hour

        education_level is used as a proxy for energy efficiency/awareness:
        higher education_level => LOWER consumption (more efficient appliances & behavior).
        """
        consumption = np.zeros(8760)

        # --- 1) Map education_level -> efficiency factors (higher => lower) ---
        level = (education_level or "university").lower().strip()

        # Simple, explainable factors. Tune later if needed.
        factors = {
            "basic": {
                "base_load": 0.18,        # higher standby/older appliances
                "prob_mult": 1.05,        # slightly more frequent peaks
                "peak_mult": 1.10,        # bigger peaks (less efficient appliances)
                "occ_low": 0.35,          # higher background load when home
                "occ_high": 0.95,
                "wash_per_week": 3.5,
            },
            "high_school": {
                "base_load": 0.15,
                "prob_mult": 1.00,
                "peak_mult": 1.00,
                "occ_low": 0.30,
                "occ_high": 0.80,
                "wash_per_week": 3.0,
            },
            "university": {
                "base_load": 0.12,        # lower standby/efficient appliances
                "prob_mult": 0.95,        # slightly less frequent peaks
                "peak_mult": 0.90,        # smaller peaks
                "occ_low": 0.22,
                "occ_high": 0.60,
                "wash_per_week": 2.7,
            },
        }
        f = factors.get(level, factors["university"])

        # --- 2) Base load (always-on devices: router, fridge standby, etc.) ---
        consumption += f["base_load"]

        # --- 3) Stochastic appliance usage when people are home ---
        for hour in range(8760):
            if household_profile[hour] == 1:
                hod = hour % 24

                # Morning (6-10): kettle, coffee, breakfast
                if hod in range(6, 10):
                    if np.random.random() < 0.3 * f["prob_mult"]:
                        consumption[hour] += np.random.uniform(1.5, 2.5) * f["peak_mult"]

                # Midday (11-14): cooking
                if hod in range(11, 14):
                    if np.random.random() < 0.2 * f["prob_mult"]:
                        consumption[hour] += np.random.uniform(1.0, 2.0) * f["peak_mult"]

                # Evening (17-22): cooking, TV, lighting, appliances
                if hod in range(17, 22):
                    if np.random.random() < 0.4 * f["prob_mult"]:
                        consumption[hour] += np.random.uniform(2.0, 3.5) * f["peak_mult"]

                # General occupancy load (background)
                consumption[hour] += np.random.uniform(f["occ_low"], f["occ_high"])

        # --- 4) Washing machine (few times per week) ---
        days_per_year = 365
        wash_days_count = int(days_per_year * (f["wash_per_week"] / 7.0))
        wash_days_count = max(0, min(days_per_year, wash_days_count))

        if wash_days_count > 0:
            washing_days = np.random.choice(days_per_year, size=wash_days_count, replace=False)
            for day in washing_days:
                wash_hour = day * 24 + np.random.randint(8, 20)
                if wash_hour < 8760:
                    consumption[wash_hour:wash_hour+2] += 1.5 * f["peak_mult"]  # 1.5 kW for 2 hours

        annual_consumption = np.sum(consumption)
        print(f"Annual consumption (appliances): {annual_consumption:.0f} kWh (education_level={level})")

        return consumption

    def add_ev_consumption(self, consumption: np.ndarray, 
                          battery_capacity: float, 
                          annual_km: float) -> np.ndarray:
        """Add electric vehicle charging to consumption profile"""
        if battery_capacity == 0 or annual_km == 0:
            return consumption
        
        # Average consumption: 18 kWh / 100 km
        ev_consumption_per_km = 0.18
        annual_ev_energy = annual_km * ev_consumption_per_km
        
        # Distribute charging events throughout year
        # Assume charging every 3 days, evening hours
        charging_days = int(365 / 3)
        charge_per_session = annual_ev_energy / charging_days
        
        ev_consumption = consumption.copy()
        
        for i in range(charging_days):
            day = int(i * 365 / charging_days)
            charge_start_hour = day * 24 + np.random.randint(18, 22)
            
            if charge_start_hour < 8760:
                # Charging for several hours
                charge_hours = int(np.ceil(charge_per_session / 7.0))  # 7 kW charger
                for h in range(min(charge_hours, 8760 - charge_start_hour)):
                    ev_consumption[charge_start_hour + h] += 7.0
        
        ev_annual = np.sum(ev_consumption) - np.sum(consumption)
        print(f"Annual EV consumption: {ev_annual:.0f} kWh")
        
        return ev_consumption
    
    def calculate_pv_production(self, 
                                installed_power: float,  # kWp
                                location: Location,
                                orientation: str = 'south',
                                tilt: float = 35) -> np.ndarray:
        """
        Calculate hourly PV production for the year
        
        Args:
            installed_power: Installed PV power in kWp
            location: Location with climate data
            orientation: Roof orientation
            tilt: Panel tilt angle in degrees
        
        Returns:
            Array of 8760 hourly production values in kWh
        """
        # Simplified PV model
        # Real model would use detailed radiation calculations
        
        production = np.zeros(8760)
        
        for hour in range(8760):
            # Get solar radiation (W/m²)
            radiation = location.global_radiation[hour]
            
            # Temperature effect
            temp = location.temperature[hour]
            temp_coefficient = 1.0 - 0.004 * (temp - 25)  # -0.4% per °C above 25°C
            
            # Orientation factor (simplified)
            orientation_factor = {
                'south': 1.0,
                'southeast': 0.95,
                'southwest': 0.95,
                'east': 0.85,
                'west': 0.85,
                'north': 0.6
            }.get(orientation.lower(), 1.0)
            
            # Calculate production
            # Standard: 1 kWp produces ~1 kWh from 1000 W/m² radiation
            efficiency = 0.18  # 18% panel efficiency
            system_losses = 0.85  # 15% system losses
            
            production[hour] = (
                installed_power *  # kWp
                (radiation / 1000) *  # Normalize to STC
                orientation_factor *
                temp_coefficient *
                system_losses
            )
        
        annual_production = np.sum(production)
        print(f"Annual PV production ({installed_power} kWp): {annual_production:.0f} kWh")
        
        return production
    
    def calculate_energy_balance(self,
                                consumption: np.ndarray,
                                production: np.ndarray,
                                battery_capacity: float = 0) -> Dict:
        """
        Calculate energy balance with or without battery
        
        Returns:
            Dictionary with:
            - self_consumption: kWh consumed directly from PV
            - grid_export: kWh exported to grid
            - grid_import: kWh imported from grid
            - battery_cycles: Number of full battery cycles
        """
        self_consumption = np.zeros(8760)
        grid_export = np.zeros(8760)
        grid_import = np.zeros(8760)
        
        battery_state = 0  # kWh
        battery_charge = np.zeros(8760)
        battery_discharge = np.zeros(8760)
        
        for hour in range(8760):
            pv = production[hour]
            load = consumption[hour]
            
            # Direct consumption
            direct_use = min(pv, load)
            self_consumption[hour] = direct_use
            
            surplus = pv - direct_use
            deficit = load - direct_use
            
            if battery_capacity > 0:
                # Try to charge battery with surplus
                if surplus > 0:
                    charge = min(surplus, battery_capacity - battery_state, battery_capacity * 0.5)  # Max 0.5C charge rate
                    battery_state += charge * 0.95  # 95% charging efficiency
                    battery_charge[hour] = charge
                    surplus -= charge
                
                # Try to discharge battery for deficit
                if deficit > 0:
                    discharge = min(deficit, battery_state, battery_capacity * 0.5)  # Max 0.5C discharge rate
                    battery_state -= discharge
                    battery_discharge[hour] = discharge * 0.95  # 95% discharge efficiency
                    self_consumption[hour] += battery_discharge[hour]
                    deficit -= battery_discharge[hour]
            
            # Remaining surplus goes to grid
            grid_export[hour] = surplus
            
            # Remaining deficit comes from grid
            grid_import[hour] = deficit
        
        total_battery_throughput = np.sum(battery_discharge)
        battery_cycles = total_battery_throughput / battery_capacity if battery_capacity > 0 else 0
        
        results = {
            'self_consumption_kwh': np.sum(self_consumption),
            'grid_export_kwh': np.sum(grid_export),
            'grid_import_kwh': np.sum(grid_import),
            'battery_cycles': battery_cycles,
            'self_consumption_rate': np.sum(self_consumption) / np.sum(production) * 100 if np.sum(production) > 0 else 0,
            'autarky_rate': np.sum(self_consumption) / np.sum(consumption) * 100 if np.sum(consumption) > 0 else 0
        }
        
        return results
    
    def calculate_economics(self,
                           energy_balance: Dict,
                           installed_power: float,
                           battery_capacity: float,
                           electricity_price: float = 4.5,  # CZK/kWh
                           feed_in_tariff: float = 1.5,  # CZK/kWh
                           pv_cost_per_kwp: float = 22000,  # CZK
                           pv_fixed_cost: float = 120000,  # CZK (fixní náklady FVE: montáž, střídač, projekt, ...)
                           battery_cost_per_kwh: float = 9000,  # CZK
                           subsidy: float = 100000,  # CZK
                           lifetime: int = 25,  # years
                           discount_rate: float = 0.05) -> Dict:
        """
        Calculate economic metrics:
        - NPV (Net Present Value)
        - IRR (Internal Rate of Return)
        - Payback period
        """
        # Investment costs
        pv_cost = float(pv_fixed_cost) + installed_power * pv_cost_per_kwp
        battery_cost = battery_capacity * battery_cost_per_kwh
        total_cost = pv_cost + battery_cost - subsidy
        
        # Annual savings
        grid_savings = energy_balance['self_consumption_kwh'] * electricity_price
        feed_in_revenue = energy_balance['grid_export_kwh'] * feed_in_tariff
        annual_savings = grid_savings + feed_in_revenue
        
        # Calculate NPV
        npv = -total_cost
        for year in range(1, lifetime + 1):
            # Degradation: 0.5% per year
            degradation_factor = (1 - 0.005) ** year
            annual_cash_flow = annual_savings * degradation_factor
            npv += annual_cash_flow / ((1 + discount_rate) ** year)
        
        # Simple payback period
        payback_period = total_cost / annual_savings if annual_savings > 0 else float('inf')
        
        # IRR (Internal Rate of Return) computed from yearly cash-flows
        # Cash-flow convention: year 0 is the investment (negative), years 1..lifetime are annual net benefits.
        cash_flows = [-total_cost]
        for year in range(1, lifetime + 1):
            degradation_factor = (1 - 0.005) ** year
            cash_flows.append(annual_savings * degradation_factor)

        def _npv(rate: float) -> float:
            return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

        def _irr_bisection() -> float:
            # Find rate where NPV(rate)=0. Use bisection with expanding bounds.
            lo, hi = -0.9, 1.0
            f_lo, f_hi = _npv(lo), _npv(hi)

            # Expand upper bound if needed (up to 5x = 500%).
            while f_lo * f_hi > 0 and hi < 5.0:
                hi *= 1.5
                f_hi = _npv(hi)

            if f_lo * f_hi > 0:
                return float('nan')

            for _ in range(80):
                mid = (lo + hi) / 2
                f_mid = _npv(mid)
                if abs(f_mid) < 1e-6:
                    return mid
                if f_lo * f_mid <= 0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid
            return (lo + hi) / 2

        irr_rate = _irr_bisection()
        # irr_rate is a decimal (e.g., 0.12 = 12%); expose percent for UI.
        irr_percent = irr_rate * 100 if np.isfinite(irr_rate) else float("nan")

        results = {
            'pv_cost': pv_cost,
            'battery_cost': battery_cost,
            'total_cost': total_cost,
            'total_cost_with_subsidy': total_cost,
            'annual_savings': annual_savings,
            'npv': npv,
            'irr': irr_percent,  # percent (e.g., 12.3)
            'payback_period': payback_period,
            # Recommended if payback <= 15 years and IRR >= 5%
            'recommended': (payback_period <= 15) and (np.isfinite(irr_rate) and (irr_rate >= 0.05)),
        }
        
        return results
    
    def optimize_pv_size(self,
                        consumption: np.ndarray,
                        location: Location,
                        min_power: float = 1,
                        max_power: float = 15,
                        step: float = 0.5,
                        electricity_price: float = 4.5,
                        feed_in_tariff: float = 1.5) -> Tuple[float, float, Dict]:
        """
        Find optimal PV system size by maximizing NPV
        
        Returns:
            (optimal_pv_power, optimal_battery_capacity, best_economics)
        """
        best_npv = -float('inf')
        best_pv_power = min_power
        best_battery_capacity = 0
        best_results = None
        
        print("\nOptimizing PV system size...")
        
        # Grid search
        for pv_power in np.arange(min_power, max_power + step, step):
            for battery_cap in [0, pv_power * 0.5, pv_power]:  # Test without and with battery
                # Calculate production
                production = self.calculate_pv_production(
                    installed_power=pv_power,
                    location=location
                )
                
                # Calculate energy balance
                energy_balance = self.calculate_energy_balance(
                    consumption=consumption,
                    production=production,
                    battery_capacity=battery_cap
                )
                
                # Calculate economics
                economics = self.calculate_economics(
                    energy_balance=energy_balance,
                    installed_power=pv_power,
                    battery_capacity=battery_cap,
                    electricity_price=float(electricity_price),
                    feed_in_tariff=float(feed_in_tariff),
                )
                
                # Check if this is better
                if economics['npv'] > best_npv and np.isfinite(economics.get('irr', float('nan'))) and economics['irr'] >= 5:  # IRR >= 5% constraint
                    best_npv = economics['npv']
                    best_pv_power = pv_power
                    best_battery_capacity = battery_cap
                    best_results = {
                        'energy_balance': energy_balance,
                        'economics': economics
                    }
        
        print(f"Optimal PV size: {best_pv_power} kWp")
        print(f"Optimal battery: {best_battery_capacity} kWh")
        print(f"NPV: {best_npv:,.0f} CZK")
        
        return best_pv_power, best_battery_capacity, best_results


def main():
    """Example usage of the application"""
    
    # Initialize model
    model = HouseholdModel()
    
    print("\n" + "="*80)
    print("SCENARIO: 3-person household in Prague with electric vehicle")
    print("="*80)
    
    # Step 1 & 2: Add household members
    model.add_user(gender='female', age=44, status='classic_8_16')  # Mother
    model.add_user(gender='male', age=46, status='morning_shift')  # Father
    model.add_user(gender='male', age=17, status='student')  # Son
    
    # Step 3: Generate household occupation profile
    household_profile = model.generate_household_profile()
    
    # Step 4: Generate consumption profile
    base_consumption = model.generate_consumption_profile(
        household_profile=household_profile,
        education_level='university'
    )
    
    # Step 5: Add electric vehicle
    total_consumption = model.add_ev_consumption(
        consumption=base_consumption,
        battery_capacity=77,  # kWh (Škoda Enyaq)
        annual_km=15000
    )
    
    print(f"\nTotal annual consumption: {np.sum(total_consumption):.0f} kWh")
    
    # Step 6: Select location (Prague)
    location = list(model.climate_data.values())[0]
    
    # Step 7: Optimize PV system size
    optimal_pv, optimal_battery, results = model.optimize_pv_size(
        consumption=total_consumption,
        location=location,
        min_power=1,
        max_power=15,
        step=1.0
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if results:
        eb = results['energy_balance']
        ec = results['economics']
        
        print(f"\nEnergy Balance:")
        print(f"  Self-consumption: {eb['self_consumption_kwh']:,.0f} kWh ({eb['self_consumption_rate']:.1f}%)")
        print(f"  Grid export: {eb['grid_export_kwh']:,.0f} kWh")
        print(f"  Grid import: {eb['grid_import_kwh']:,.0f} kWh")
        print(f"  Autarky rate: {eb['autarky_rate']:.1f}%")
        
        print(f"\nEconomics:")
        print(f"  Investment: {ec['total_cost_with_subsidy']:,.0f} CZK (with subsidy)")
        print(f"  Annual savings: {ec['annual_savings']:,.0f} CZK")
        print(f"  Payback period: {ec['payback_period']:.1f} years")
        print(f"  NPV (25 years): {ec['npv']:,.0f} CZK")
        print(f"  IRR: {ec['irr']:.1f}%")
        print(f"  Recommendation: {'✓ RECOMMENDED' if ec['recommended'] else '✗ NOT RECOMMENDED'}")


if __name__ == '__main__':
    main()
