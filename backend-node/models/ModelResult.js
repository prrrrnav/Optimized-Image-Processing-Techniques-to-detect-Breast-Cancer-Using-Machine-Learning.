const mongoose = require("mongoose");
const resultSchema = new mongoose.Schema({
  result_id: { type: String, required: true, unique: true },
  user_id:   { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  image_id:  { type: String, ref: "ImageMetadata" },
  prediction_label: { type: String },
  confidence_score: { type: Number },
  timestamp: { type: Date, default: Date.now },
  accuracy:  { type: Number }
});

module.exports = mongoose.model("ModelResult", resultSchema);
