from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)

MONGO_URI = "mongodb+srv://shnee2505_db_user:yuvas@cluster0.pxwhd1y.mongodb.net/?appName=Cluster0"

DB_NAME = "punchdb"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
punch_collection = db["punch_logs"]

@app.route("/api/punch", methods=["POST"])
def punch_in():
    data = request.json

    punch_data = {
        "employeeId": data.get("employeeId"),
        "password": data.get("password"),
        "address": data.get("address"),
        "street": data.get("street"),
        "area": data.get("area"),
        "city": data.get("city"),
        "state": data.get("state"),
        "country": data.get("country"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "status": data.get("status"),
        "punchTime": datetime.utcnow()
    }

    punch_collection.insert_one(punch_data)

    return jsonify({"message": "Punch-in stored successfully"}), 201

if __name__ == "__main__":
    app.run(port=8000, debug=True)

