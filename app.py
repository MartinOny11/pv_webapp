"""
Flask Web Application for Photovoltaic System Dimensioning
Web UI for the 6-step process from the diploma thesis
"""

from flask import Flask, render_template, request, jsonify, session, send_file
import os
import json
import numpy as np
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from pv_dimensioning_app import HouseholdModel, User

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model globally
DATA_PATH = os.environ.get('DATA_PATH', '/mnt/user-data/uploads')
model = HouseholdModel(data_path=DATA_PATH)


@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/app')
def application():
    """Main application page"""
    # Initialize session if needed
    if 'step' not in session:
        session['step'] = 1
        session['users'] = []
        session['appliances'] = []
        session['house'] = {}
        session['location'] = None
        session['ev'] = {}
    
    return render_template('app.html', current_step=session.get('step', 1))


@app.route('/api/step1', methods=['POST'])
def step1_user_data():
    """Step 1: Basic user data - add primary user"""
    data = request.json
    
    gender = data.get('gender')
    age = int(data.get('age'))
    status = data.get('status')
    
    # Create user and determine profile
    user = User(gender=gender, age=age, status=status)
    
    # Initialize or reset session data
    session['users'] = [{
        'gender': gender,
        'age': age,
        'status': status,
        'profile_id': user.profile_id,
        'is_primary': True
    }]
    
    session['step'] = 2
    session.modified = True
    
    return jsonify({
        'success': True,
        'profile_id': user.profile_id,
        'message': f'Primary user added: Profile {user.profile_id}',
        'next_step': 2
    })


@app.route('/api/step2', methods=['POST'])
def step2_household_members():
    """Step 2: Add other household members"""
    data = request.json
    
    if data.get('action') == 'add_member':
        gender = data.get('gender')
        age = int(data.get('age'))
        status = data.get('status')
        
        user = User(gender=gender, age=age, status=status)
        
        session['users'].append({
            'gender': gender,
            'age': age,
            'status': status,
            'profile_id': user.profile_id,
            'is_primary': False
        })
        session.modified = True
        
        return jsonify({
            'success': True,
            'profile_id': user.profile_id,
            'total_members': len(session['users']),
            'message': f'Member added: Profile {user.profile_id}'
        })
    
    elif data.get('action') == 'complete':
        session['step'] = 3
        session.modified = True
        
        return jsonify({
            'success': True,
            'total_members': len(session['users']),
            'next_step': 3
        })
    
    elif data.get('action') == 'remove_member':
        index = data.get('index')
        if 0 <= index < len(session['users']):
            removed = session['users'].pop(index)
            session.modified = True
            return jsonify({
                'success': True,
                'removed': removed,
                'total_members': len(session['users'])
            })


@app.route('/api/step3', methods=['POST'])
def step3_appliances():
    """Step 3: Electrical appliances - simplified or custom selection"""
    data = request.json
    
    if data.get('action') == 'quick_estimate':
        # Quick estimate based on education level
        education_level = data.get('education_level', 'university')
        
        session['appliances'] = {
            'method': 'quick_estimate',
            'education_level': education_level
        }
        session['step'] = 4
        session.modified = True
        
        return jsonify({
            'success': True,
            'method': 'quick_estimate',
            'next_step': 4
        })
    
    elif data.get('action') == 'custom_selection':
        # Custom appliance selection
        appliances = data.get('appliances', [])
        
        session['appliances'] = {
            'method': 'custom',
            'selected': appliances
        }
        session['step'] = 4
        session.modified = True
        
        return jsonify({
            'success': True,
            'method': 'custom',
            'appliance_count': len(appliances),
            'next_step': 4
        })


@app.route('/api/step4', methods=['POST'])
def step4_location():
    """Step 4: Climate data - select location"""
    data = request.json
    
    location_name = data.get('location')
    
    if location_name in model.climate_data:
        session['location'] = location_name
        session['step'] = 5
        session.modified = True
        
        return jsonify({
            'success': True,
            'location': location_name,
            'next_step': 5
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Location {location_name} not found'
        }), 400


@app.route('/api/step5', methods=['POST'])
def step5_house():
    """Step 5: House technical data"""
    data = request.json
    
    if data.get('method') == 'predefined':
        # Use predefined house profile
        profile_id = int(data.get('profile_id'))
        
        session['house'] = {
            'method': 'predefined',
            'profile_id': profile_id,
            'floors': data.get('floors'),
            'year': data.get('year'),
            'floor_area': data.get('floor_area')
        }
        
    elif data.get('method') == 'custom':
        # Custom house parameters
        session['house'] = {
            'method': 'custom',
            'floor_area': float(data.get('floor_area')),
            'construction_year': int(data.get('construction_year')),
            'roof_type': data.get('roof_type'),
            'roof_orientation': data.get('roof_orientation'),
            'roof_slope': float(data.get('roof_slope', 35)),
            'wall_material': data.get('wall_material'),
            'wall_thickness': float(data.get('wall_thickness', 0.4))
        }
    
    session['step'] = 6
    session.modified = True
    
    return jsonify({
        'success': True,
        'next_step': 6
    })


@app.route('/api/step6', methods=['POST'])
def step6_electric_vehicle():
    """Step 6: Electric vehicle data (optional)"""
    data = request.json
    
    has_ev = data.get('has_ev', False)
    
    if has_ev:
        session['ev'] = {
            'has_ev': True,
            'battery_capacity': float(data.get('battery_capacity', 0)),
            'annual_km': float(data.get('annual_km', 0)),
            'count': int(data.get('count', 1))
        }
    else:
        session['ev'] = {'has_ev': False}
    
    session['step'] = 'calculate'
    session.modified = True
    
    return jsonify({
        'success': True,
        'next_step': 'calculate'
    })


@app.route('/api/calculate', methods=['POST'])
def calculate_results():
    """Calculate optimal PV system based on all inputs"""
    
    try:
        # Recreate household model from session
        temp_model = HouseholdModel(data_path=DATA_PATH)
        
        # Add users
        for user_data in session.get('users', []):
            temp_model.add_user(
                gender=user_data['gender'],
                age=user_data['age'],
                status=user_data['status']
            )
        
        # Generate household profile
        household_profile = temp_model.generate_household_profile()
        occupancy_rate = np.mean(household_profile) * 100
        
        # Generate consumption profile
        appliance_config = session.get('appliances', {})
        education_level = appliance_config.get('education_level', 'university')
        consumption = temp_model.generate_consumption_profile(
            household_profile=household_profile,
            education_level=education_level
        )
        
        base_consumption = np.sum(consumption)
        
        # Add EV consumption if applicable
        ev_config = session.get('ev', {})
        if ev_config.get('has_ev'):
            ev_count = ev_config.get('count', 1)
            for _ in range(ev_count):
                consumption = temp_model.add_ev_consumption(
                    consumption=consumption,
                    battery_capacity=ev_config.get('battery_capacity', 0),
                    annual_km=ev_config.get('annual_km', 0)
                )
        
        total_consumption = np.sum(consumption)
        ev_consumption = total_consumption - base_consumption
        
        # Get location
        location_name = session.get('location')
        if not location_name or location_name not in temp_model.climate_data:
            location = list(temp_model.climate_data.values())[0]
            location_name = location.name
        else:
            location = temp_model.climate_data[location_name]
        
        # Optimize PV system
        optimal_pv, optimal_battery, results = temp_model.optimize_pv_size(
            consumption=consumption,
            location=location,
            min_power=1,
            max_power=15,
            step=1.0
        )
        
        # Prepare results for display
        if results:
            eb = results['energy_balance']
            ec = results['economics']
            
            response_data = {
                'success': True,
                'household': {
                    'member_count': len(session.get('users', [])),
                    'occupancy_rate': round(occupancy_rate, 1),
                    'location': location_name
                },
                'consumption': {
                    'base': round(base_consumption, 0),
                    'ev': round(ev_consumption, 0),
                    'total': round(total_consumption, 0)
                },
                'optimal_system': {
                    'pv_power': optimal_pv,
                    'battery_capacity': optimal_battery
                },
                'energy_balance': {
                    'self_consumption': round(eb['self_consumption_kwh'], 0),
                    'grid_export': round(eb['grid_export_kwh'], 0),
                    'grid_import': round(eb['grid_import_kwh'], 0),
                    'self_consumption_rate': round(eb['self_consumption_rate'], 1),
                    'autarky_rate': round(eb['autarky_rate'], 1),
                    'battery_cycles': round(eb.get('battery_cycles', 0), 1)
                },
                'economics': {
                    'pv_cost': round(ec['pv_cost'], 0),
                    'battery_cost': round(ec['battery_cost'], 0),
                    'total_cost': round(ec['total_cost'], 0),
                    'total_cost_with_subsidy': round(ec['total_cost_with_subsidy'], 0),
                    'annual_savings': round(ec['annual_savings'], 0),
                    'payback_period': round(ec['payback_period'], 1),
                    'npv': round(ec['npv'], 0),
                    'irr': round(ec['irr'], 1),
                    'recommended': ec['recommended']
                }
            }
            
            # Store results in session for later retrieval
            session['results'] = response_data
            session.modified = True
            
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': 'Could not find optimal solution. Try adjusting parameters.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of available locations"""
    locations = sorted(list(model.climate_data.keys()))
    return jsonify({
        'success': True,
        'locations': locations
    })


@app.route('/api/house-profiles', methods=['GET'])
def get_house_profiles():
    """Get list of available house profiles"""
    profiles = []
    for profile_id in sorted(model.house_profiles.keys())[:50]:
        # Map profile IDs to descriptions based on thesis
        # Simplified mapping - would be more detailed in production
        floor_area_ranges = ['do 80 m²', '81-100 m²', '101-120 m²', '121-150 m²', 'nad 151 m²']
        year_ranges = ['<1994', '1995-2002', '2003-2011', '2012-2018', '>2018']
        floors = [1, 2]
        
        idx = profile_id - 1
        floor_idx = idx % 2
        year_idx = (idx // 2) % 5
        area_idx = (idx // 10) % 5
        
        profiles.append({
            'id': profile_id,
            'floors': floors[floor_idx],
            'year_range': year_ranges[year_idx],
            'floor_area': floor_area_ranges[area_idx],
            'description': f'{floors[floor_idx]} podlaží, {year_ranges[year_idx]}, {floor_area_ranges[area_idx]}'
        })
    
    return jsonify({
        'success': True,
        'profiles': profiles[:50]
    })


@app.route('/api/session', methods=['GET'])
def get_session():
    """Get current session state"""
    return jsonify({
        'step': session.get('step', 1),
        'users': session.get('users', []),
        'appliances': session.get('appliances', {}),
        'location': session.get('location'),
        'house': session.get('house', {}),
        'ev': session.get('ev', {})
    })


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session and start over"""
    session.clear()
    session['step'] = 1
    session['users'] = []
    session['appliances'] = []
    session['house'] = {}
    session['location'] = None
    session['ev'] = {}
    session.modified = True
    
    return jsonify({
        'success': True,
        'message': 'Session reset'
    })


@app.route('/api/chart/<chart_type>', methods=['GET'])
def generate_chart(chart_type):
    """Generate charts for visualization"""
    
    if 'results' not in session:
        return jsonify({'error': 'No results available'}), 404
    
    results = session['results']
    
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'energy_balance':
        # Energy balance pie chart
        eb = results['energy_balance']
        labels = ['Self-consumption', 'Grid Export', 'Grid Import']
        sizes = [eb['self_consumption'], eb['grid_export'], eb['grid_import']]
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Annual Energy Balance')
        plt.axis('equal')
        
    elif chart_type == 'economics':
        # Economics bar chart
        ec = results['economics']
        categories = ['Investment', 'Annual\nSavings\n(x10)', 'NPV\n(÷1000)']
        values = [
            ec['total_cost_with_subsidy'],
            ec['annual_savings'] * 10,
            ec['npv'] / 1000
        ]
        colors_bar = ['#F44336', '#4CAF50', '#2196F3']
        
        plt.bar(categories, values, color=colors_bar)
        plt.title('Economic Overview')
        plt.ylabel('CZK')
        plt.grid(axis='y', alpha=0.3)
        
    elif chart_type == 'consumption':
        # Consumption breakdown
        cons = results['consumption']
        labels = ['Base\nConsumption', 'EV\nConsumption']
        sizes = [cons['base'], cons['ev']]
        colors = ['#2196F3', '#FF9800']
        
        if cons['ev'] > 0:
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            plt.text(0.5, 0.5, f"Total: {cons['total']:,.0f} kWh/year\n(No EV)", 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
        
        plt.title('Annual Consumption Breakdown')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')


@app.route('/api/export/pdf', methods=['GET'])
def export_pdf():
    """Export results as PDF report"""
    # This would require reportlab or similar library
    # Placeholder for now
    return jsonify({
        'success': False,
        'message': 'PDF export feature coming soon'
    })


@app.route('/api/export/json', methods=['GET'])
def export_json():
    """Export results as JSON"""
    if 'results' not in session:
        return jsonify({'error': 'No results available'}), 404
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'users': session.get('users', []),
            'appliances': session.get('appliances', {}),
            'location': session.get('location'),
            'house': session.get('house', {}),
            'ev': session.get('ev', {})
        },
        'results': session['results']
    }
    
    # Create response
    response = app.response_class(
        response=json.dumps(export_data, indent=2),
        status=200,
        mimetype='application/json'
    )
    response.headers["Content-Disposition"] = "attachment; filename=pv_results.json"
    
    return response


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
