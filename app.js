const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const cors = require("cors");

const Employee = require("./models/Employee");

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect("mongodb://127.0.0.1:27017/wfh_attendance")
  .then(() => console.log("MongoDB Connected"))
  .catch(err => console.error(err));

// Punch schema
const punchSchema = new mongoose.Schema({
    employeeId: { type: String, unique: true }, // Only one record per employee
    latitude: Number,
    longitude: Number,
    address: String,
    status: String,
    punchedAt: { type: Date, default: Date.now }
});
const Punch = mongoose.model("Punch", punchSchema);

// Punch-in API with authentication
app.post("/api/punch-in", async (req, res) => {
    try {
        const { employeeId, password, latitude, longitude, address, status } = req.body;

        // Authenticate employee
        const employee = await Employee.findOne({ employeeId });
        if (!employee) return res.status(401).json({ message: "Invalid Employee ID or password" });

        const isMatch = await employee.comparePassword(password);
        if (!isMatch) return res.status(401).json({ message: "Invalid Employee ID or password" });

        // Update existing punch or create new if not exists
        const punch = await Punch.findOneAndUpdate(
            { employeeId },
            { latitude, longitude, address, status, punchedAt: new Date() },
            { new: true, upsert: true } // upsert: create if not exists
        );

        console.log("Punch saved successfully:", punch);
        res.json({ message: "Punch-in successful" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Server error" });
    }
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));

