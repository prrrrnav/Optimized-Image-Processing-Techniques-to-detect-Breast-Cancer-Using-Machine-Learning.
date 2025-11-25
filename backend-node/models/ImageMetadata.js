const mongoose = require("mongoose");
const imageMeta = new mongoose.Schema({
  image_id:   { type: String, required: true, unique: true },
  user_id:    { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  file_path:  { type: String, required: true },
  upload_date:{ type: Date, default: Date.now }
});

module.exports = mongoose.model("ImageMetadata", imageMeta);
