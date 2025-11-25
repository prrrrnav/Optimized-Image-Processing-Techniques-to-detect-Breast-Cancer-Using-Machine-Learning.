const express = require("express");
const router = express.Router();
const User = require("../models/User");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

const JWT_SECRET = process.env.JWT_SECRET || "replace_with_a_strong_secret";

// signup
router.post("/signup", async (req, res) => {
  try {
    const { username, email, password } = req.body;
    if (!email || !password || !username) return res.status(400).json({ error: "Missing fields" });

    if (await User.findOne({ email })) return res.status(400).json({ error: "Email already exists" });

    const hash = await bcrypt.hash(password, 10);
    const user = new User({ username, email, password: hash });
    await user.save();
    return res.json({ message: "User created", user: { id: user._id, email: user.email, username: user.username } });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "server" });
  }
});

// login
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Missing fields" });

    const user = await User.findOne({ email });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const ok = await bcrypt.compare(password, user.password);
    if (!ok) return res.status(401).json({ error: "Invalid credentials" });

    const token = jwt.sign({ uid: user._id, email: user.email }, JWT_SECRET, { expiresIn: "12h" });
    return res.json({ token, user: { id: user._id, email: user.email, username: user.username } });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "server" });
  }
});

module.exports = router;
