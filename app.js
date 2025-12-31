const express = require("express");
const mongoose = require("mongoose");

const app = express();
app.use(express.json());
app.use(express.static(__dirname)); // serves your existing HTML

// MongoDB connection
mongoose.connect("mongodb://127.0.0.1:27017/employee_tracking");

const EmployeeSchema = new mongoose.Schema({
    employeeId: { type: String, unique: true },
    password: { type: String, required: true },
    latitude: Number,
    longitude: Number,
    updatedAt: { type: Date, default: Date.now }
});

const Employee = mongoose.model("Employee", EmployeeSchema);

// Punch-In API
app.post("/punch-in", async (req, res) => {
    const { employeeId, password, latitude, longitude } = req.body;

    if (!employeeId || !password) {
        return res.json({ success: false, message: "Employee ID and password are required" });
    }

   const emp = await Employee.findOne({ employeeId });

if (!emp) {
    return res.json({ success: false, message: "Employee not found" });
}
if (emp.password !== password) {
    return res.json({ success: false, message: "Incorrect password" });
}


    await Employee.updateOne(
        { employeeId },
        { latitude, longitude, updatedAt: new Date() }
    );

    res.json({ success: true });
});
app.post("/update-location", async (req, res) => {
    const { employeeId, password, latitude, longitude } = req.body;

    const emp = await Employee.findOne({ employeeId });

    if (!emp || emp.password !== password) {
        return res.json({ success: false, message: "Unauthorized" });
    }

    await Employee.updateOne(
        { employeeId },
        { latitude, longitude, updatedAt: new Date() }
    );

    res.json({ success: true });
});


app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
});
