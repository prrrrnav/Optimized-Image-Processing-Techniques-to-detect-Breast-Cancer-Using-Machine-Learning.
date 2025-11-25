const mongoose = require("mongoose");
const logSchema = new mongoose.Schema({
  action_type: String,
  user_id: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  date_time: { type: Date, default: Date.now },
  details: mongoose.Schema.Types.Mixed
});
module.exports = mongoose.model("SystemLog", logSchema);
