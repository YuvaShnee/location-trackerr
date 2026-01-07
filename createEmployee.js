const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const Employee = require("./models/Employee");

const MONGO_URI =
  "mongodb+srv://2324046_db_user:kaviya2602@cluster0.lamp0vs.mongodb.net/wfh_attendance?retryWrites=true&w=majority";

async function createEmployee() {
  try {
    // Connect to MongoDB
    await mongoose.connect(MONGO_URI);
    console.log(" MongoDB connected");

    const employeeId = "EMP-1234";
    const plainPassword = "kaka234";

    // Check if employee already exists
    const existing = await Employee.findOne({ employeeId });
    if (existing) {
      console.log(" Employee already exists. Delete it first if needed.");
      process.exit(0);
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(plainPassword, 10);

    // Create employee
    const employee = new Employee({
      employeeId,
      password: hashedPassword,
      name: "Kaviya",
      email: "kaviya@example.com",
      salary: 50000
    });

    await employee.save();

    console.log(" Employee created successfully");
    console.log(" Login credentials:");
    console.log(" Employee ID:", employeeId);
    console.log(" Password:", plainPassword);

    process.exit(0);
  } catch (error) {
    console.error(" Error creating employee:", error);
    process.exit(1);
  }
}

createEmployee();
