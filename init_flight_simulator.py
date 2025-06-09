import sqlite3
from datetime import datetime, timedelta
import os
import random

# Paths
SCHEDULE_DB = "flight_status.db"
GEO_DB = "geo_tracking.db"
WEATHER_DB = "weather.db"

# Constants
BASE_AIRPORT = "JFK"
BASE_LAT, BASE_LON = 40.6413, -73.7781
AIRLINES = [
    "American Airlines", "United Airlines", "Delta", "Southwest",
    "British Airways", "Air France", "Lufthansa", "Emirates",
    "Qatar Airways", "Turkish Airlines"
]
DESTINATIONS = [
    "ATL", "DFW", "ORD", "MIA", "SEA", "LAX", "CDG", "FRA", "DXB", "DOH"
]

def simple_distance_km(lat1, lon1, lat2, lon2):
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    return ((delta_lat ** 2 + delta_lon ** 2) ** 0.5) * 111

def generate_phase(i, now):
    if i == 0:
        departure = now - timedelta(hours=1, minutes=15)
        arrival = now + timedelta(minutes=15)
        phase = "DEVIATED"
        actual_lat = BASE_LAT + 0.6
        actual_lon = BASE_LON + 0.6
        altitude = random.randint(20000, 38000)
    elif i == 1:
        departure = now - timedelta(minutes=45)
        arrival = now + timedelta(minutes=30)
        phase = "IN_AIR"
        actual_lat = BASE_LAT + 0.2
        actual_lon = BASE_LON + 0.1
        altitude = random.randint(20000, 40000)
    elif i == 2:
        departure = now + timedelta(minutes=10)
        arrival = now + timedelta(hours=2)
        phase = "HOLD_SHORT"
        actual_lat = BASE_LAT + 0.002
        actual_lon = BASE_LON + 0.001
        altitude = 0
    else:
        departure = now + timedelta(hours=1 + i * 0.1)
        arrival = departure + timedelta(hours=2)
        phase = "GATE"
        actual_lat = BASE_LAT
        actual_lon = BASE_LON
        altitude = 0
    return departure, arrival, phase, actual_lat, actual_lon, altitude

def simulate_weather_at(lat, lon):
    return {
        "lat": lat,
        "lon": lon,
        "wind_speed_kt": random.randint(5, 40),
        "visibility_km": round(random.uniform(1, 10), 1),
        "precip_mm": round(random.uniform(0, 10), 1),
        "storm": int(random.random() < 0.3),
        "fog": int(random.random() < 0.2),
        "temperature_c": round(random.uniform(-5, 35), 1)
    }

def compute_weather_risk(w):
    risk = 0
    if w["wind_speed_kt"] > 30: risk += 2
    if w["visibility_km"] < 5: risk += 3
    if w["precip_mm"] > 5: risk += 2
    if w["storm"]: risk += 3
    if w["fog"]: risk += 2
    return min(risk, 10)

def init_all_dbs():
    now = datetime.utcnow()

    for db in [SCHEDULE_DB, GEO_DB, WEATHER_DB]:
        if os.path.exists(db):
            os.remove(db)

    # Schedule DB
    conn_sched = sqlite3.connect(SCHEDULE_DB)
    cur_sched = conn_sched.cursor()
    cur_sched.execute("CREATE TABLE flights (flight_id TEXT PRIMARY KEY, airline TEXT, origin TEXT, destination TEXT, sched_departure TEXT, sched_arrival TEXT, status TEXT, gate TEXT)")

    # Geo DB
    conn_geo = sqlite3.connect(GEO_DB)
    cur_geo = conn_geo.cursor()
    cur_geo.execute("""
        CREATE TABLE geotracking (
            flight_id TEXT PRIMARY KEY,
            expected_lat REAL,
            expected_lon REAL,
            actual_lat REAL,
            actual_lon REAL,
            altitude_ft INTEGER,
            distance_to_dest_km REAL,
            heading_deg INTEGER,
            phase TEXT,
            last_updated TEXT
        )
    """)

    # Weather DB
    conn_weather = sqlite3.connect(WEATHER_DB)
    cur_weather = conn_weather.cursor()
    cur_weather.execute("""
        CREATE TABLE weather_by_flight (
            flight_id TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            wind_speed_kt INTEGER,
            visibility_km REAL,
            precip_mm REAL,
            storm INTEGER,
            fog INTEGER,
            temperature_c REAL,
            risk_score INTEGER
        )
    """)

    for i, airline in enumerate(AIRLINES):
        flight_id = f"{airline[:2].upper()}00{i+1}"
        destination = DESTINATIONS[i % len(DESTINATIONS)]
        gate = f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 10)}"
        sched_departure, sched_arrival, phase, actual_lat, actual_lon, altitude = generate_phase(i, now)

        cur_sched.execute("INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            flight_id, airline, BASE_AIRPORT, destination,
            sched_departure.isoformat(), sched_arrival.isoformat(), "SCHEDULED", gate
        ))

        distance = simple_distance_km(actual_lat, actual_lon, BASE_LAT, BASE_LON)
        heading = random.randint(0, 359)

        cur_geo.execute("INSERT INTO geotracking VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            flight_id, BASE_LAT, BASE_LON, actual_lat, actual_lon, altitude,
            round(distance, 2), heading, phase, now.isoformat()
        ))

        weather = simulate_weather_at(actual_lat, actual_lon)
        risk_score = compute_weather_risk(weather)
        cur_weather.execute("INSERT INTO weather_by_flight VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            flight_id, weather["lat"], weather["lon"], weather["wind_speed_kt"],
            weather["visibility_km"], weather["precip_mm"], weather["storm"],
            weather["fog"], weather["temperature_c"], risk_score
        ))

    conn_sched.commit(); conn_sched.close()
    conn_geo.commit(); conn_geo.close()
    conn_weather.commit(); conn_weather.close()

    return "✅ Initialized all 3 DBs with weather risk scoring."

init_all_dbs()
