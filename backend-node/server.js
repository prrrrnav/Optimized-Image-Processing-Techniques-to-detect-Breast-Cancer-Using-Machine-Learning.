// server.js — Node backend with JWT auth, image hosting, ML forwarding, history
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const FormData = require("form-data");
const cors = require("cors");
const axios = require("axios");
const mongoose = require("mongoose");
const { v4: uuidv4 } = require("uuid");

const app = express();
app.use(cors());
app.use(express.json()); // for auth routes

// --- ENV
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/image_demo";
const PYTHON_INFER_URL = process.env.PYTHON_INFER_URL || "http://127.0.0.1:5001/infer";
const PORT = process.env.PORT || 3000;

// --- Connect to MongoDB (modern)
mongoose.connect(MONGO_URI)
  .then(()=> console.log("MongoDB connected"))
  .catch(err => console.error("MongoDB connection error:", err));

// --- Models & routes
const User = require("./models/User");
const ImageMetadata = require("./models/ImageMetadata");
const ModelResult = require("./models/ModelResult");

const authRoutes = require("./routes/auth");
const authMiddleware = require("./middleware/auth");

app.use("/api/auth", authRoutes);

// --- Upload dir & static hosting
const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);
app.use("/uploads", express.static(UPLOAD_DIR));

// --- Multer
const upload = multer({ dest: UPLOAD_DIR });

// --- Health
app.get("/health", (req, res) => res.json({ status: "ok" }));

// --- Protected predict route (requires JWT)
app.post("/predict", authMiddleware, upload.single("image"), async (req, res) => {
  console.log("=== /predict called ===");
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const userId = req.user.id;
    const imageId = uuidv4();
    const filePath = req.file.path;
    const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // Save image metadata
    await ImageMetadata.create({ image_id: imageId, user_id: userId, file_path: filePath, upload_date: new Date() });

    // Forward to Python ML server
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));

    const headers = form.getHeaders();
    // forward auth header if needed
    headers["Authorization"] = req.headers.authorization || "";

    const mlResp = await axios.post(PYTHON_INFER_URL, form, {
      headers: { ...headers, Accept: "application/json" },
      maxContentLength: Infinity, maxBodyLength: Infinity, timeout: 30000,
    });

    const mlData = mlResp.data || {};
    const resultId = uuidv4();

    // Save result
    await ModelResult.create({
      result_id: resultId,
      user_id: userId,
      image_id: imageId,
      prediction_label: mlData.prediction || mlData.pred,
      confidence_score: parseFloat(mlData.confidence || mlData.conf || 0),
      timestamp: new Date(),
      accuracy: mlData.test_accuracy || null
    });

    // Respond with ml data plus IDs and URL
    return res.json({ ...mlData, result_id: resultId, image_id: imageId, image_url: fileUrl });
  } catch (err) {
    console.error("Forwarding error:", err && err.toString());
    if (err.response) {
      console.error("ML status:", err.response.status);
      console.error("ML response data:", err.response.data);
    }
    if (req.file && req.file.path) try { fs.unlinkSync(req.file.path); } catch(e){}
    return res.status(500).json({ error: "Server error", detail: err.message });
  }
});

// --- History endpoint (protected)
app.get("/api/history", authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const page = Math.max(1, parseInt(req.query.page || "1"));
    const perPage = Math.min(100, parseInt(req.query.perPage || "20"));
    const skip = (page - 1) * perPage;

    const items = await ModelResult.find({ user_id: userId }).sort({ timestamp: -1 }).skip(skip).limit(perPage).lean();
    const total = await ModelResult.countDocuments({ user_id: userId });
    return res.json({ items, page, perPage, total });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "server" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Node backend running on http://localhost:${PORT}`);
});