const express = require("express");
const router = express.Router();
const ModelResult = require("../models/ModelResult");
const auth = require("../middleware/auth");

// get paginated history for the logged-in user
router.get("/", auth, async (req,res)=>{
  const page = parseInt(req.query.page||"1");
  const perPage = Math.min(parseInt(req.query.perPage||"20"), 100);
  const skip = (page-1)*perPage;
  const userId = req.user.id;

  const items = await ModelResult.find({ user_id: userId })
    .sort({ timestamp: -1 })
    .skip(skip).limit(perPage)
    .lean();

  const total = await ModelResult.countDocuments({ user_id: userId });
  res.json({ items, page, perPage, total });
});

// get a single result (only if it belongs to user)
router.get("/:result_id", auth, async (req,res)=>{
  const r = await ModelResult.findOne({ result_id: req.params.result_id, user_id: req.user.id }).lean();
  if(!r) return res.status(404).json({error:"Not found"});
  res.json(r);
});

module.exports = router;
