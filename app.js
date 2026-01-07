const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const cors = require("cors");

const Employee = require("./models/Employee");


const app = express();
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(
  "mongodb+srv://2324046_db_user:kaviya2602@cluster0.lamp0vs.mongodb.net/wfh_attendance?retryWrites=true&w=majority"
)
.then(() => console.log("MongoDB Atlas Connected"))
.catch(err => console.error("MongoDB Atlas Error:", err));

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
    console.log("📥 Request received:", req.body);

    const { employeeId, password, latitude, longitude, address, status } = req.body;

    const employee = await Employee.findOne({ employeeId });

    console.log("👤 Employee found:", employee);

    if (!employee) {
      return res.status(401).json({ message: "Invalid Employee ID or password" });
    }

    const isMatch = await employee.comparePassword(password);

    if (!isMatch) {
      return res.status(401).json({ message: "Invalid Employee ID or password" });
    }

    const punch = await Punch.findOneAndUpdate(
      { employeeId },
      { latitude, longitude, address, status, punchedAt: new Date() },
      { upsert: true, new: true }
    );

    console.log("✅ Punch stored:", punch);

    res.status(200).json({ message: "Punch-in successful", punch });

  } catch (err) {
    console.error("🔥 Server error:", err);
    res.status(500).json({ message: "Server error" });
  }
});


app.listen(3000, () => console.log("Server running on http://localhost:3000"));
