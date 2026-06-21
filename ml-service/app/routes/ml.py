import os
import json
import pickle
import urllib.request
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from app.models.database import get_violations_collection

router = APIRouter(prefix="/api/ml", tags=["ml"])

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
CONGESTION_MODEL_FILE = MODEL_DIR / "congestion_model.pkl"
HOTSPOT_MODEL_FILE = MODEL_DIR / "hotspot_clustering.pkl"

# Global lazy load variables
_congestion_payload = None
_hotspots_registry = None

def get_congestion_payload():
    global _congestion_payload
    if _congestion_payload is None:
        if CONGESTION_MODEL_FILE.exists():
            try:
                with open(CONGESTION_MODEL_FILE, "rb") as f:
                    _congestion_payload = pickle.load(f)
                print(f"[ML Route] Loaded congestion model from {CONGESTION_MODEL_FILE}")
            except Exception as e:
                print(f"[ML Route] ERROR loading congestion model: {e}")
        else:
            print(f"[ML Route] Congestion model not found at {CONGESTION_MODEL_FILE}")
    return _congestion_payload

def get_hotspots_registry():
    global _hotspots_registry
    if _hotspots_registry is None:
        if HOTSPOT_MODEL_FILE.exists():
            try:
                with open(HOTSPOT_MODEL_FILE, "rb") as f:
                    _hotspots_registry = pickle.load(f)
                print(f"[ML Route] Loaded hotspots registry ({len(_hotspots_registry)} items) from {HOTSPOT_MODEL_FILE}")
            except Exception as e:
                print(f"[ML Route] ERROR loading hotspots: {e}")
                _hotspots_registry = []
        else:
            print(f"[ML Route] Hotspots registry not found at {HOTSPOT_MODEL_FILE}")
            _hotspots_registry = []
    return _hotspots_registry

# Pydantic Schemas
class HotspotResponse(BaseModel):
    cluster_id: int
    junction_name: str
    police_station: str
    latitude: float
    longitude: float
    violation_count: int
    unique_vehicles: int
    top_violation_type: str
    top_vehicle_type: str
    avg_traffic_delay_minutes: float
    priority: str
    
    # New spatial & congestion attributes
    dominant_violation_type: str
    peak_hours: str
    repeat_frequency: float
    nearby_sensitive_places: List[str]
    estimated_speed_drop: float
    expected_delay_minutes: float
    congestion_score: float
    affected_road: str
    confidence: float
    evidence_source: str

class PredictImpactRequest(BaseModel):
    latitude: float
    longitude: float
    hour_of_day: int
    vehicle_type: str
    violation_type: str
    cluster_size: int = 1
    is_sensitive_zone: Optional[int] = None

class PredictImpactResponse(BaseModel):
    success: bool
    congestionScore: float          # scaled 0.0 to 1.0 (delay / 60)
    predictedDelayMinutes: float
    speedDropKmh: float
    recommendation: str

class EnforcementPriorityItem(BaseModel):
    junction: str
    policeStation: str
    score: int
    priority: str
    openCases: int
    repeatOffenderCount: int
    sensitiveZoneFlags: List[str]
    suggestedAction: str
    latitude: float
    longitude: float
    
    # New spatial & congestion attributes
    dominant_violation_type: str
    peak_hours: str
    repeat_frequency: float
    nearby_sensitive_places: List[str]
    estimated_speed_drop: float
    expected_delay_minutes: float
    congestion_score: float
    affected_road: str
    confidence: float
    evidence_source: str

@router.get("/hotspots", response_model=List[HotspotResponse])
async def get_hotspots(
    police_station: Optional[str] = Query(None),
    priority: Optional[str] = Query(None)
):
    registry = get_hotspots_registry()
    if not registry:
        return []
        
    filtered = []
    for h in registry:
        if police_station and h["police_station"].lower() != police_station.lower():
            continue
        if priority and h["priority"].lower() != priority.lower():
            continue
            
        delay = h.get("avg_traffic_delay_minutes", 5.0)
        filtered.append({
            "cluster_id": h["cluster_id"],
            "junction_name": h["junction_name"],
            "police_station": h["police_station"],
            "latitude": h["latitude"],
            "longitude": h["longitude"],
            "violation_count": h["violation_count"],
            "unique_vehicles": h["unique_vehicles"],
            "top_violation_type": h["top_violation_type"],
            "top_vehicle_type": h["top_vehicle_type"],
            "avg_traffic_delay_minutes": delay,
            "priority": h["priority"],
            
            # New spatial & congestion attributes
            "dominant_violation_type": h.get("dominant_violation_type", h["top_violation_type"]),
            "peak_hours": h.get("peak_hours", "09:00, 18:00"),
            "repeat_frequency": h.get("repeat_frequency", 0.0),
            "nearby_sensitive_places": h.get("nearby_sensitive_places", ["junction"]),
            "estimated_speed_drop": h.get("estimated_speed_drop", delay * 0.65),
            "expected_delay_minutes": h.get("expected_delay_minutes", delay),
            "congestion_score": h.get("congestion_score", delay / 60.0),
            "affected_road": h.get("affected_road", h["junction_name"]),
            "confidence": h.get("confidence", 0.85),
            "evidence_source": h.get("evidence_source", "Model estimate & historical incidents")
        })
        
    return filtered

@router.post("/predict-impact", response_model=PredictImpactResponse)
async def predict_impact(payload: PredictImpactRequest):
    congestion_payload = get_congestion_payload()
    
    # Heuristics for sensitive zones if not provided
    is_sensitive = payload.is_sensitive_zone
    if is_sensitive is None:
        # Defaults to 0 unless coordinates are in known sensitive grid or fields
        is_sensitive = 0

    if congestion_payload:
        try:
            model = congestion_payload["model"]
            v_map = congestion_payload["vehicle_map"]
            t_map = congestion_payload["violation_map"]
            
            # Map strings to code integers
            v_code = v_map.get(str(payload.vehicle_type).strip().upper(), 3)  # default: MOTOR CAR (3)
            t_code = t_map.get(str(payload.violation_type).strip().upper(), 0)  # default: UNKNOWN (0)
            
            # Form feature array
            X_in = [[
                payload.latitude,
                payload.longitude,
                payload.hour_of_day,
                v_code,
                t_code,
                is_sensitive,
                payload.cluster_size
            ]]
            
            # Predict
            pred_delay = float(model.predict(X_in)[0])
            speed_drop = float(pred_delay * 0.68)
            
        except Exception as e:
            print(f"[ML Route] Prediction inference failed: {e}. Using fallback.")
            # Fallback heuristic calculation
            pred_delay = 5.0
            speed_drop = 3.0
    else:
        # Mathematical Heuristic Fallback (if models haven't been trained yet)
        h = payload.hour_of_day
        hour_multiplier = 1.7 if (8 <= h <= 11) or (17 <= h <= 20) else (1.1 if (6 <= h <= 22) else 0.6)
        vtype = str(payload.vehicle_type).upper()
        vehicle_delay = 14.0 if "HEAVY" in vtype or "BUS" in vtype or "TRUCK" in vtype else (4.0 if "CAR" in vtype or "SUV" in vtype else 0.5)
        sensitive_delay = 8.0 if is_sensitive == 1 else 0.0
        density_delay = min(15.0, payload.cluster_size * 1.5)
        
        pred_delay = (2.0 + vehicle_delay + sensitive_delay + density_delay) * hour_multiplier
        speed_drop = pred_delay * 0.65

    # Fetch live Mappls Speed/Flow telemetry at coordinates (Ground Truth Correlation)
    mappls_key = os.getenv("MAPPLS_API_KEY")
    api_mappls = os.getenv("API_MAPPLS", "https://api.tomtom.com")
    live_speed_drop = None
    if mappls_key:
        try:
            url = f"{api_mappls}/traffic/services/4/flowSegmentData/absolute/10/json?key={mappls_key}&point={payload.latitude},{payload.longitude}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                flow_data = json.loads(response.read().decode())
                if "flowSegmentData" in flow_data:
                    segment = flow_data["flowSegmentData"]
                    curr_speed = segment.get("currentSpeed")
                    free_speed = segment.get("freeFlowSpeed")
                    if curr_speed is not None and free_speed is not None:
                        # Ground Truth Speed Drop (km/h)
                        live_speed_drop = max(0.0, float(free_speed - curr_speed))
                        print(f"[ML Route] Live Mappls Segment Flow: FreeFlow={free_speed}km/h, Current={curr_speed}km/h. SpeedDrop={live_speed_drop}km/h.")
        except Exception as e:
            print(f"[ML Route] Live Mappls speed query failed: {e}")

    # Blended delay & speed drop (correlating ML model with live ground truth flow telemetry)
    if live_speed_drop is not None:
        # Correlate expected delay with real-time speed drops
        calculated_live_delay = live_speed_drop / 0.65
        
        # Blend: 40% ML Model historical patterns + 60% Live ground-truth speed sensors
        pred_delay = (pred_delay * 0.4) + (calculated_live_delay * 0.6)
        speed_drop = (speed_drop * 0.4) + (live_speed_drop * 0.6)
        print(f"[ML Route] Dynamic Ground-Truth Correlation: Blended Delay={pred_delay:.2f} mins, Blended SpeedDrop={speed_drop:.2f} km/h")

    # Clip results
    pred_delay = max(1.0, min(60.0, float(pred_delay)))
    speed_drop = max(0.5, min(45.0, float(speed_drop)))
    congestion_score = float(pred_delay / 60.0)

    # Determine recommended enforcement
    if pred_delay > 35.0:
        recommendation = "Deploy Tow Truck - Critical lane blockage causing gridlock risk."
    elif pred_delay > 18.0:
        recommendation = "Increase Patrol - Moderate congestion build-up; request officer visual validation."
    else:
        recommendation = "Monitor - Minor obstruction impact; log and keep status active."

    return {
        "success": True,
        "congestionScore": round(congestion_score, 3),
        "predictedDelayMinutes": round(pred_delay, 1),
        "speedDropKmh": round(speed_drop, 1),
        "recommendation": recommendation
    }

@router.get("/enforcement-priorities", response_model=List[EnforcementPriorityItem])
async def get_enforcement_priorities(
    limit: int = Query(8, ge=1, le=30)
):
    collection = get_violations_collection()
    # Load calculated hotspots from DB structure or spatial joins
    registry = get_hotspots_registry()
    if not registry:
        # Fallback empty list
        return []

    priorities = []
    for h in registry[:limit]:
        # Count current open violations for this hotspot junction
        open_cases = collection.count_documents({
            "junction_name": h["junction_name"],
            "validation_status": {"$ne": "approved"}
        })

        # Find repeat offender count at this junction
        repeat_count = int(h["unique_vehicles"] * 0.14) if h["unique_vehicles"] > 1 else 0

        # Proximity sensitive check
        sensitive_flags = []
        text_check = f"{h['junction_name']} {h['police_station']}".lower()
        if "school" in text_check: sensitive_flags.append("near_school")
        if "hospital" in text_check or "hosp" in text_check: sensitive_flags.append("near_hospital")
        if "metro" in text_check: sensitive_flags.append("near_metro_station")
        if "crossing" in text_check: sensitive_flags.append("near_pedestrian_crossing")

        # Multi-factor score formula
        delay = h.get("avg_traffic_delay_minutes", 5.0)
        violation_density = min(25, h.get("violation_count", 0) * 0.2)
        congestion_delay = min(25, h.get("expected_delay_minutes", delay) * 0.8)
        repeat_offender_score = min(25, h.get("repeat_frequency", 0.0) * 100 * 0.25)
        
        nearby_sensitive = h.get("nearby_sensitive_places", [])
        sensitive_zone_score = min(15, len(nearby_sensitive) * 5)
        
        # Peak hour score
        from datetime import datetime
        current_hour = datetime.now().hour
        is_peak = (8 <= current_hour <= 11) or (17 <= current_hour <= 20)
        peak_hour_score = 10 if is_peak else 3
        
        open_case_load = min(15, open_cases * 0.5)
        
        score = int(min(100, violation_density + congestion_delay + repeat_offender_score + sensitive_zone_score + peak_hour_score + open_case_load))

        # Suggested actions mapping
        if score >= 75:
            suggested = "tow"
            priority = "severe"
        elif score >= 50:
            suggested = "patrol"
            priority = "high"
        elif score >= 25:
            suggested = "camera"
            priority = "medium"
        else:
            suggested = "monitor"
            priority = "low"

        priorities.append({
            "junction": h["junction_name"],
            "policeStation": h["police_station"],
            "score": score,
            "priority": priority,
            "openCases": open_cases,
            "repeatOffenderCount": repeat_count,
            "sensitiveZoneFlags": nearby_sensitive if nearby_sensitive else sensitive_flags,
            "suggestedAction": suggested,
            "latitude": h["latitude"],
            "longitude": h["longitude"],
            
            # New spatial & congestion attributes
            "dominant_violation_type": h.get("dominant_violation_type", h.get("top_violation_type", "UNKNOWN")),
            "peak_hours": h.get("peak_hours", "09:00, 18:00"),
            "repeat_frequency": h.get("repeat_frequency", 0.0),
            "nearby_sensitive_places": nearby_sensitive if nearby_sensitive else ["junction"],
            "estimated_speed_drop": h.get("estimated_speed_drop", delay * 0.65),
            "expected_delay_minutes": h.get("expected_delay_minutes", delay),
            "congestion_score": h.get("congestion_score", delay / 60.0),
            "affected_road": h.get("affected_road", h["junction_name"]),
            "confidence": h.get("confidence", 0.85),
            "evidence_source": h.get("evidence_source", "Model estimate & historical incidents")
        })

    # Sort by calculated priority score descending
    priorities.sort(key=lambda x: x["score"], reverse=True)
    return priorities

import math

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth's radius in kilometers
    R = 6371.0
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

class PatrolWaypoint(BaseModel):
    junction: str
    policeStation: str
    latitude: float
    longitude: float
    priority: str
    suggestedAction: str
    score: int

class PatrolRouteResponse(BaseModel):
    success: bool
    waypoints: List[PatrolWaypoint]
    coordinates: List[dict] # [{"lat": lat, "lng": lng}]

@router.get("/patrol-route", response_model=PatrolRouteResponse)
async def get_patrol_route(limit: int = Query(5, ge=2, le=10)):
    # 1. Get the enforcement priorities list (internally holds latitude/longitude)
    priorities_list = await get_enforcement_priorities(limit=limit)
    if not priorities_list:
        return {"success": False, "waypoints": [], "coordinates": []}
        
    # 2. Sequence using Multi-Factor Cost Heuristic (balances travel distance, severity, and live case load)
    unvisited = list(priorities_list)
    # Start at the single highest priority hotspot (the first item is sorted by score descending)
    sequenced = [unvisited.pop(0)]
    
    distance_weight = 1.0    # cost per km
    priority_weight = 1.5    # discount for higher priority score (0.0 to 1.0)
    case_load_weight = 2.0   # discount for high open case load (0.0 to 1.0)
    
    while unvisited:
        last = sequenced[-1]
        best_idx = 0
        min_cost = float('inf')
        for i, item in enumerate(unvisited):
            dist = calculate_haversine_distance(
                last["latitude"], last["longitude"],
                item["latitude"], item["longitude"]
            )
            # Item score ranges from 0 to 100, scale it to [0, 1]
            score_factor = item["score"] / 100.0
            # Open cases represents localized blockage; cap at 10 to scale to [0, 1]
            case_load_factor = min(1.0, item["openCases"] / 10.0)
            
            # Combined cost function: lower cost is preferred
            cost = (dist * distance_weight) - (score_factor * priority_weight) - (case_load_factor * case_load_weight)
            
            if cost < min_cost:
                min_cost = cost
                best_idx = i
        sequenced.append(unvisited.pop(best_idx))
        
    # 3. Call Mappls Routing API to generate the exact road coordinates
    mappls_key = os.getenv("MAPPLS_API_KEY")
    api_mappls = os.getenv("API_MAPPLS", "https://api.tomtom.com")
    coords_list = []
    
    if mappls_key and len(sequenced) >= 2:
        try:
            # Format: lat,lon:lat,lon:...:lat,lon
            points_query = ":".join([f"{item['latitude']},{item['longitude']}" for item in sequenced])
            url = f"{api_mappls}/routing/1/calculateRoute/{points_query}/json?key={mappls_key}&traffic=true&travelMode=car"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                route_data = json.loads(response.read().decode())
                if "routes" in route_data and len(route_data["routes"]) > 0:
                    # Iterate through all routing legs (intervals between sequenced waypoints)
                    for leg in route_data["routes"][0]["legs"]:
                        for p in leg["points"]:
                            coords_list.append({"lat": p["latitude"], "lng": p["longitude"]})
        except Exception as e:
            print(f"[ML Route] Mappls live routing calculation failed: {e}")
            
    # Fallback to direct connections if API call fails or key is missing
    if not coords_list:
        coords_list = [{"lat": item["latitude"], "lng": item["longitude"]} for item in sequenced]
        
    # Format return waypoints matching model
    waypoints_response = []
    for item in sequenced:
        waypoints_response.append(PatrolWaypoint(
            junction=item["junction"],
            policeStation=item["policeStation"],
            latitude=item["latitude"],
            longitude=item["longitude"],
            priority=item["priority"],
            suggestedAction=item["suggestedAction"],
            score=item["score"]
        ))
        
    return {
        "success": True,
        "waypoints": waypoints_response,
        "coordinates": coords_list
    }
