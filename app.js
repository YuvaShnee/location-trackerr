const express = require("express");
const mongoose = require("mongoose");

const app = express();
app.use(express.json());
app.use(express.static(__dirname)); // serves your existing HTML

// MongoDB connection
mongoose.connect("mongodb://127.0.0.1:27017/employee_tracking");

const EmployeeSchema = new mongoose.Schema({
    employeeId: { type: String, unique: true },
    latitude: Number,
    longitude: Number,
    updatedAt: { type: Date, default: Date.now }
});

const Employee = mongoose.model("Employee", EmployeeSchema);

// Punch-In API
app.post("/punch-in", async (req, res) => {
    const { employeeId, latitude, longitude } = req.body;

    await Employee.findOneAndUpdate(
        { employeeId },
        { employeeId, latitude, longitude, updatedAt: new Date() },
        { upsert: true }
    );

    res.json({ success: true });
});

// Live location update every 8 sec
app.post("/update-location", async (req, res) => {
    const { employeeId, latitude, longitude } = req.body;

    await Employee.updateOne(
        { employeeId },
        { latitude, longitude, updatedAt: new Date() }
    );

    res.json({ success: true });
});

app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
});
