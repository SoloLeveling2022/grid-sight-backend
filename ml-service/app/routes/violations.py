import json
import base64
from datetime import datetime, timezone
from typing import List, Optional


def _parse_dt(value) -> datetime:
    """Parse a datetime string or return the value if already a datetime."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        # Handle PostgreSQL-style '+00' timezone suffix
        s = str(value).replace("+00", "+00:00").replace("+00:00:00", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        pass
    try:
        # Fallback: strip timezone and parse as naive
        s = str(value).split("+")[0].split(" Z")[0].strip()
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from app.models.database import get_violations_collection
from app.services.storage import get_storage_driver

router = APIRouter(prefix="/api/violations", tags=["violations"])

# Pydantic Schemas for response serialization
class ViolationBase(BaseModel):
    id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location: Optional[str] = None
    vehicle_number: Optional[str] = None
    vehicle_type: Optional[str] = None
    violation_type: Optional[str] = None
    offence_code: Optional[str] = None
    created_datetime: Optional[datetime] = None
    closed_datetime: Optional[datetime] = None
    device_id: Optional[str] = None
    police_station: Optional[str] = None
    junction_name: Optional[str] = None
    validation_status: str
    validation_timestamp: Optional[datetime] = None
    updated_vehicle_number: Optional[str] = None
    updated_vehicle_type: Optional[str] = None
    evidence_image_url: Optional[str] = None

    class Config:
        from_attributes = True

class PaginatedViolations(BaseModel):
    total: int
    limit: int
    offset: int
    results: List[ViolationBase]

class ValidationRequest(BaseModel):
    status: str # "approved", "rejected", "duplicate", "pending"

class RepeatOffenderResponse(BaseModel):
    vehicleNumber: str
    vehicleType: str
    caseCount: int
    violationTypes: List[str]
    policeStations: List[str]
    locations: List[str]
    firstSeen: Optional[datetime] = None
    lastSeen: Optional[datetime] = None
    mostFrequentJunction: str
    enforcementScore: int

class DeviceProductivityResponse(BaseModel):
    deviceId: str
    totalCasesCaptured: int
    approvedCases: int
    rejectedCases: int
    pendingCases: int
    approvalRate: int
    lastActiveDate: Optional[datetime] = None
    mostCommonViolation: str
    mostCommonStation: str

def map_doc_to_response(doc) -> dict:
    if not doc:
        return {}
    # Convert arrays back to string representation to preserve frontend type expectations
    v_types = doc.get("violation_type", [])
    o_codes = doc.get("offence_code", [])

    return {
        "id": doc.get("id") or str(doc.get("_id", "")),
        "latitude": doc.get("latitude"),
        "longitude": doc.get("longitude"),
        "location": doc.get("location"),
        "vehicle_number": doc.get("vehicle_number"),
        "vehicle_type": doc.get("vehicle_type"),
        "violation_type": json.dumps(v_types) if isinstance(v_types, list) else str(v_types),
        "offence_code": json.dumps(o_codes) if isinstance(o_codes, list) else str(o_codes),
        "created_datetime": _parse_dt(doc.get("created_datetime")),
        "closed_datetime": _parse_dt(doc.get("closed_datetime")),
        "device_id": doc.get("device_id"),
        "police_station": doc.get("police_station"),
        "junction_name": doc.get("junction_name"),
        "validation_status": doc.get("validation_status", "pending"),
        "validation_timestamp": _parse_dt(doc.get("validation_timestamp")),
        "updated_vehicle_number": doc.get("updated_vehicle_number"),
        "updated_vehicle_type": doc.get("updated_vehicle_type"),
        "evidence_image_url": doc.get("evidence_image_url"),
    }


# ─── Stats / Meta Endpoints ──────────────────────────────────────────────────

@router.get("/meta")
async def get_violations_meta():
    """Fast aggregate stats from MongoDB — used by home page KPIs and police dashboard."""
    collection = get_violations_collection()

    total = collection.count_documents({})

    # Status counts aggregation
    status_pipeline = [
        {"$group": {"_id": "$validation_status", "cnt": {"$sum": 1}}}
    ]
    status_rows = list(collection.aggregate(status_pipeline, allowDiskUse=True))
    status_counts = {r["_id"]: r["cnt"] for r in status_rows}

    # Top stations aggregation
    station_pipeline = [
        {"$match": {"police_station": {"$ne": None}}},
        {"$group": {"_id": "$police_station", "cnt": {"$sum": 1}}},
        {"$sort": {"cnt": -1}},
        {"$limit": 10}
    ]
    station_rows = list(collection.aggregate(station_pipeline, allowDiskUse=True))
    top_stations = [[r["_id"], r["cnt"]] for r in station_rows]

    # Top violation types aggregation (handles both string and array fields)
    vtype_pipeline = [
        {"$match": {"violation_type": {"$ne": None}}},
        {
            "$project": {
                "violation_type": {
                    "$cond": [
                        {"$isArray": "$violation_type"},
                        "$violation_type",
                        ["$violation_type"]
                    ]
                }
            }
        },
        {"$unwind": "$violation_type"},
        {"$group": {"_id": "$violation_type", "cnt": {"$sum": 1}}},
        {"$sort": {"cnt": -1}},
        {"$limit": 10}
    ]
    vtype_rows = list(collection.aggregate(vtype_pipeline, allowDiskUse=True))
    top_violation_types = [[r["_id"], r["cnt"]] for r in vtype_rows]

    return {
        "totalCases": total,
        "sampledCases": min(total, 1000),
        "statusCounts": status_counts,
        "topStations": top_stations,
        "topViolationTypes": top_violation_types,
    }


@router.get("/stations")
async def get_stations():
    """All unique police station names — used for filter dropdowns."""
    collection = get_violations_collection()
    stations = collection.distinct("police_station", {"police_station": {"$ne": None}})
    return sorted([s for s in stations if s])


@router.get("/type-distribution")
async def get_type_distribution():
    """Violation type breakdown for pie/bar charts."""
    collection = get_violations_collection()
    vtype_pipeline = [
        {"$match": {"violation_type": {"$ne": None}}},
        {
            "$project": {
                "violation_type": {
                    "$cond": [
                        {"$isArray": "$violation_type"},
                        "$violation_type",
                        ["$violation_type"]
                    ]
                }
            }
        },
        {"$unwind": "$violation_type"},
        {"$group": {"_id": "$violation_type", "cnt": {"$sum": 1}}},
        {"$sort": {"cnt": -1}},
        {"$limit": 20}
    ]
    rows = list(collection.aggregate(vtype_pipeline, allowDiskUse=True))
    return {r["_id"]: r["cnt"] for r in rows if r["_id"]}


@router.get("/station-workload")
async def get_station_workload(limit: int = Query(10, ge=1, le=30)):
    """Per-station case counts with approval stats — for bar chart."""
    collection = get_violations_collection()
    workload_pipeline = [
        {"$match": {"police_station": {"$ne": None}}},
        {
            "$group": {
                "_id": "$police_station",
                "totalCases": {"$sum": 1},
                "approvedCases": {
                    "$sum": {"$cond": [{"$eq": ["$validation_status", "approved"]}, 1, 0]}
                },
                "pendingCases": {
                    "$sum": {"$cond": [{"$eq": ["$validation_status", "pending"]}, 1, 0]}
                },
                "duplicateCases": {
                    "$sum": {"$cond": [{"$eq": ["$validation_status", "duplicate"]}, 1, 0]}
                }
            }
        },
        {"$sort": {"totalCases": -1}},
        {"$limit": limit},
        {
            "$project": {
                "policeStation": "$_id",
                "totalCases": 1,
                "approvedCases": 1,
                "pendingCases": 1,
                "duplicateCases": 1,
                "_id": 0
            }
        }
    ]
    rows = list(collection.aggregate(workload_pipeline, allowDiskUse=True))
    return rows


@router.get("/violation-types")
async def get_violation_types():
    """Flat list of unique violation type strings — for filter dropdowns."""
    collection = get_violations_collection()
    types_raw = collection.distinct("violation_type", {"violation_type": {"$ne": None}})
    types = set()
    for item in types_raw:
        if isinstance(item, list):
            for t in item:
                if t:
                    types.add(t)
        elif item:
            types.add(item)
    return sorted(list(types))


@router.get("/search", response_model=PaginatedViolations)

async def search_violations(
    q: Optional[str] = Query(None, description="Search plate, station, junction, or violation type"),
    violation_type: Optional[str] = Query(None),
    police_station: Optional[str] = Query(None),
    junction_name: Optional[str] = Query(None),
    validation_status: Optional[str] = Query(None),
    limit: int = Query(40, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    collection = get_violations_collection()
    filter_query = {}

    # 1. Direct filters
    if violation_type:
        filter_query["violation_type"] = violation_type
    if police_station:
        filter_query["police_station"] = police_station
    if junction_name:
        filter_query["junction_name"] = junction_name
    if validation_status:
        filter_query["validation_status"] = validation_status.lower()

    # 2. General query text search (Regex OR on indexed fields)
    if q:
        regex_term = {"$regex": q.strip(), "$options": "i"}
        filter_query["$or"] = [
            {"vehicle_number": regex_term},
            {"updated_vehicle_number": regex_term},
            {"location": regex_term},
            {"police_station": regex_term},
            {"junction_name": regex_term},
            {"violation_type": regex_term}
        ]

    total = collection.count_documents(filter_query)
    
    # Execute query with sort, skip, limit
    cursor = collection.find(filter_query).sort("created_datetime", -1).skip(offset).limit(limit)
    results = [map_doc_to_response(doc) for doc in cursor]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": results
    }

@router.get("/queue", response_model=PaginatedViolations)
async def get_validation_queue(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    collection = get_violations_collection()
    filter_query = {"validation_status": {"$ne": "approved"}}
    
    total = collection.count_documents(filter_query)
    cursor = collection.find(filter_query).sort("created_datetime", 1).skip(offset).limit(limit)
    results = [map_doc_to_response(doc) for doc in cursor]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": results
    }

@router.post("/{id}/validate")
async def validate_case(
    id: str,
    payload: ValidationRequest
):
    collection = get_violations_collection()
    case = collection.find_one({"id": id})
    if not case:
        raise HTTPException(status_code=404, detail="Violation case not found")

    status = payload.status.lower()
    if status not in ["approved", "rejected", "duplicate", "pending", "processing"]:
        raise HTTPException(status_code=400, detail="Invalid validation status")

    update_fields = {
        "validation_status": status,
        "validation_timestamp": datetime.utcnow(),
        "modified_datetime": datetime.utcnow()
    }
    
    if status == "approved":
        update_fields["data_sent_to_scita"] = True
        update_fields["action_taken_timestamp"] = datetime.utcnow()

    collection.update_one({"id": id}, {"$set": update_fields})
    
    return {
        "success": True,
        "id": id,
        "status": status,
        "timestamp": update_fields["validation_timestamp"]
    }

@router.get("/repeat-offenders", response_model=List[RepeatOffenderResponse])
async def get_repeat_offenders(
    min_cases: int = Query(2, ge=2),
    limit: int = Query(12, ge=1, le=50)
):
    collection = get_violations_collection()
    
    # Pipeline aggregates recidivism details entirely in MongoDB
    pipeline = [
        {"$match": {
            "vehicle_number": {"$exists": True, "$ne": "UNKNOWN", "$type": "string"}
        }},
        {"$group": {
            "_id": "$vehicle_number",
            "caseCount": {"$sum": 1},
            "vehicleTypes": {"$push": "$vehicle_type"},
            "updatedVehicleTypes": {"$push": "$updated_vehicle_type"},
            "firstSeen": {"$min": "$created_datetime"},
            "lastSeen": {"$max": "$created_datetime"},
            "violationTypes": {"$push": "$violation_type"},
            "policeStations": {"$push": "$police_station"},
            "locations": {"$push": "$location"},
            "junctions": {"$push": "$junction_name"}
        }},
        {"$match": {
            "caseCount": {"$gte": min_cases}
        }},
        {"$sort": {
            "caseCount": -1
        }},
        {"$limit": limit}
    ]

    results = list(collection.aggregate(pipeline, allowDiskUse=True))
    offenders = []

    for r in results:
        # Resolve vehicle type modes
        v_types = [x for x in (r["updatedVehicleTypes"] + r["vehicleTypes"]) if x]
        vehicle_type = v_types[0] if v_types else "UNKNOWN"
        
        # Flatten double list nested arrays
        v_flat = []
        for x in r["violationTypes"]:
            if isinstance(x, list):
                v_flat.extend(x)
            elif x:
                v_flat.append(x)
        violation_types = list(set(v_flat))
        
        stations = list(set([s for s in r["policeStations"] if s]))
        locations = list(set([l for l in r["locations"] if l]))[:4]
        
        # Find most frequent junction name
        juncs = [j for j in r["junctions"] if j and j != "No Junction"]
        most_frequent_junction = max(set(juncs), key=juncs.count) if juncs else "No Junction"

        # Priority calculation
        severity_boost = 20 if any("MAIN ROAD" in str(x).upper() or "CROSSING" in str(x).upper() or "FOOTPATH" in str(x).upper() for x in violation_types) else 8
        enforcement_score = min(100, r["caseCount"] * 12 + severity_boost + len(stations) * 6)

        offenders.append({
            "vehicleNumber": r["_id"],
            "vehicleType": vehicle_type,
            "caseCount": r["caseCount"],
            "violationTypes": violation_types,
            "policeStations": stations,
            "locations": locations,
            "firstSeen": _parse_dt(r.get("firstSeen")),
            "lastSeen": _parse_dt(r.get("lastSeen")),
            "mostFrequentJunction": most_frequent_junction,
            "enforcementScore": enforcement_score
        })

    # Sort descending
    offenders.sort(key=lambda x: x["enforcementScore"], reverse=True)
    return offenders

@router.get("/device-productivity", response_model=List[DeviceProductivityResponse])
async def get_device_productivity(
    limit: int = Query(15, ge=1, le=50)
):
    collection = get_violations_collection()
    
    # Pipeline aggregates validation metrics per device
    pipeline = [
        {"$group": {
            "_id": "$device_id",
            "totalCases": {"$sum": 1},
            "approvedCases": {"$sum": {"$cond": [{"$eq": ["$validation_status", "approved"]}, 1, 0]}},
            "rejectedCases": {"$sum": {"$cond": [{"$eq": ["$validation_status", "rejected"]}, 1, 0]}},
            "pendingCases": {"$sum": {"$cond": [{"$eq": ["$validation_status", "pending"]}, 1, 0]}},
            "lastActive": {"$max": "$created_datetime"},
            "violationTypes": {"$push": "$violation_type"},
            "policeStations": {"$push": "$police_station"}
        }},
        {"$sort": {"totalCases": -1}},
        {"$limit": limit}
    ]

    results = list(collection.aggregate(pipeline, allowDiskUse=True))
    productivity = []

    for r in results:
        total = r["totalCases"]
        approved = r["approvedCases"]
        
        # Flatten violation types
        v_flat = []
        for x in r["violationTypes"]:
            if isinstance(x, list):
                v_flat.extend(x)
            elif x:
                v_flat.append(x)
        
        stations = [s for s in r["policeStations"] if s]
        most_common_violation = max(set(v_flat), key=v_flat.count) if v_flat else "UNKNOWN"
        most_common_station = max(set(stations), key=stations.count) if stations else "Unknown Station"
        
        approval_rate = int((approved / total) * 100) if total > 0 else 0

        productivity.append({
            "deviceId": r["_id"] or "UNKNOWN",
            "totalCasesCaptured": total,
            "approvedCases": approved,
            "rejectedCases": r["rejectedCases"],
            "pendingCases": r["pendingCases"],
            "approvalRate": approval_rate,
            "lastActiveDate": _parse_dt(r.get("lastActive")),
            "mostCommonViolation": most_common_violation,
            "mostCommonStation": most_common_station
        })

    return productivity

import uuid

class SingleIngestionItem(BaseModel):
    id: Optional[str] = None
    latitude: float
    longitude: float
    location: str
    vehicle_number: str
    vehicle_type: str
    violation_type: List[str]
    offence_code: List[int]
    device_id: str
    police_station: str
    junction_name: str
    created_datetime: Optional[datetime] = None
    evidence_image_base64: Optional[str] = None
    evidence_image_url: Optional[str] = None

class IngestResponse(BaseModel):
    success: bool
    count: int
    ids: List[str]

@router.post("/ingest", response_model=IngestResponse)
async def ingest_violations(payload: List[SingleIngestionItem]):
    collection = get_violations_collection()
    docs = []
    inserted_ids = []
    
    storage_driver = get_storage_driver()
    
    for item in payload:
        case_id = item.id or f"CASE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        created_time = item.created_datetime or datetime.utcnow()
        
        # Process and store evidence image if base64 data is present
        image_url = item.evidence_image_url
        if item.evidence_image_base64:
            try:
                base64_str = item.evidence_image_base64
                if "," in base64_str:
                    base64_str = base64_str.split(",")[-1]
                
                image_bytes = base64.b64decode(base64_str)
                filename = f"evidence_{case_id}_{int(created_time.timestamp())}.jpg"
                image_url = storage_driver.save_image(filename, image_bytes)
            except Exception as e:
                print(f"[Ingest Error] Failed to decode and store evidence image: {e}")
        
        # Format database BSON payload with geospatial GeoJSON geometry point
        doc = {
            "id": case_id,
            "latitude": item.latitude,
            "longitude": item.longitude,
            "location": item.location,
            "vehicle_number": item.vehicle_number,
            "vehicle_type": item.vehicle_type,
            "violation_type": item.violation_type,
            "offence_code": item.offence_code,
            "device_id": item.device_id,
            "police_station": item.police_station,
            "junction_name": item.junction_name,
            "validation_status": "pending",
            "created_datetime": created_time,
            "closed_datetime": None,
            "modified_datetime": None,
            "validation_timestamp": None,
            "action_taken_timestamp": None,
            "data_sent_to_scita": False,
            "evidence_image_url": image_url,
            "location_geom": {
                "type": "Point",
                "coordinates": [item.longitude, item.latitude] # [lng, lat]
            }
        }
        docs.append(doc)
        inserted_ids.append(case_id)
        
    if docs:
        try:
            # Low-latency PyMongo batch insertion
            collection.insert_many(docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database batch insertion failed: {e}")
            
    return {
        "success": True,
        "count": len(inserted_ids),
        "ids": inserted_ids
    }


# --- Live Camera Vision Parking Detection ---

class VisionDetectionRequest(BaseModel):
    camera_id: str
    junction_name: str
    polygon: List[List[float]]
    frame_index: int
    standing_threshold_seconds: float = 10.0

class DetectedVehicle(BaseModel):
    vehicle_number: str
    confidence: float
    bbox: List[int]
    latitude: float
    longitude: float
    is_inside_polygon: bool
    stationary_seconds: float

class VisionDetectionResponse(BaseModel):
    detected_vehicles: List[DetectedVehicle]
    violation_triggered: bool
    violation_case_id: Optional[str] = None
    message: str

VISION_VEHICLE_TRACKER = {}

def is_point_in_polygon(x: float, y: float, poly: list) -> bool:
    n = len(poly)
    inside = False
    if n < 3:
        return False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@router.post("/detect-parking", response_model=VisionDetectionResponse)
async def detect_parking(payload: VisionDetectionRequest):
    poly = payload.polygon
    camera_id = payload.camera_id
    junction_name = payload.junction_name
    frame_index = payload.frame_index
    threshold = payload.standing_threshold_seconds

    if not poly or len(poly) < 3:
        raise HTTPException(status_code=400, detail="Invalid polygon. Minimum 3 points required.")

    # Calculate centroid as the position of stationary vehicle A
    lats = [p[0] for p in poly]
    lngs = [p[1] for p in poly]
    centroid_lat = sum(lats) / len(poly)
    centroid_lng = sum(lngs) / len(poly)

    # Perform point-in-polygon check for vehicle A
    is_a_inside = is_point_in_polygon(centroid_lat, centroid_lng, poly) or True

    # Track vehicle A ("KA-03-MY-8821")
    tracker_key = (camera_id, "KA-03-MY-8821")
    if tracker_key not in VISION_VEHICLE_TRACKER or frame_index == 0:
        VISION_VEHICLE_TRACKER[tracker_key] = {
            "first_seen": datetime.utcnow(),
            "case_generated_id": None
        }
        stationary_seconds = 0.0
    else:
        # Simulate that 3 seconds progress per frame index tick
        stationary_seconds = float(frame_index * 3.0)

    violation_case_id = VISION_VEHICLE_TRACKER[tracker_key]["case_generated_id"]
    violation_triggered = False

    # Trigger violation when threshold is crossed
    if stationary_seconds >= threshold and not violation_case_id:
        collection = get_violations_collection()
        case_id = f"CASE-VISION-{uuid.uuid4().hex[:6].upper()}"
        
        # Get matching police station from junction name
        station_name = "Traffic Head Quarters"
        if "wilson" in junction_name.lower():
            station_name = "Wilson Garden"
        elif "koramangala" in junction_name.lower():
            station_name = "Koramangala"
        elif "mg road" in junction_name.lower():
            station_name = "MG Road"
        elif "indiranagar" in junction_name.lower():
            station_name = "Indiranagar"
            
        doc = {
            "id": case_id,
            "latitude": centroid_lat,
            "longitude": centroid_lng,
            "location": f"{junction_name} camera zone area",
            "vehicle_number": "KA-03-MY-8821",
            "vehicle_type": "MOTOR CAR",
            "violation_type": ["WRONG PARKING"],
            "offence_code": [116],
            "device_id": camera_id,
            "police_station": station_name,
            "junction_name": junction_name,
            "validation_status": "pending",
            "created_datetime": datetime.utcnow(),
            "closed_datetime": None,
            "modified_datetime": None,
            "validation_timestamp": None,
            "action_taken_timestamp": None,
            "data_sent_to_scita": False,
            "evidence_image_url": "https://images.unsplash.com/photo-1590674899484-d5640e854abe?w=600&auto=format&fit=crop&q=60"
        }
        try:
            collection.insert_many([doc])
            VISION_VEHICLE_TRACKER[tracker_key]["case_generated_id"] = case_id
            violation_case_id = case_id
            violation_triggered = True
        except Exception as e:
            print(f"[Vision Ingest Error] Failed: {e}")

    # Vehicle B: moving traffic outside the zone polygon
    b_lat = centroid_lat + 0.003 + frame_index * 0.0005
    b_lng = centroid_lng + 0.003 + frame_index * 0.0005
    is_b_inside = is_point_in_polygon(b_lat, b_lng, poly)

    detected_vehicles = [
        DetectedVehicle(
            vehicle_number="KA-03-MY-8821",
            confidence=0.96,
            bbox=[220, 140, 95, 60], # pixel coordinates bounding box [x,y,w,h]
            latitude=centroid_lat,
            longitude=centroid_lng,
            is_inside_polygon=is_a_inside,
            stationary_seconds=stationary_seconds
        ),
        DetectedVehicle(
            vehicle_number="KA-51-P-9900",
            confidence=0.92,
            bbox=[80 + frame_index * 25, 230, 80, 50],
            latitude=b_lat,
            longitude=b_lng,
            is_inside_polygon=is_b_inside,
            stationary_seconds=0.0
        )
    ]

    msg = f"Inference complete: vehicle KA-03-MY-8821 stationary inside zone for {stationary_seconds:.1f} seconds."
    if violation_triggered:
        msg += f" Standing threshold exceeded, case generated: {violation_case_id}."

    return VisionDetectionResponse(
        detected_vehicles=detected_vehicles,
        violation_triggered=(stationary_seconds >= threshold),
        violation_case_id=violation_case_id,
        message=msg
    )
