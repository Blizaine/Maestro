// Backward-compatible alias retained for old Pinokio bookmarks and in-flight
// dev installs. The standard hardware-aware torch.js now owns the CUDA 13/Sol
// runtime so Install, Update, Repair, and Start cannot drift apart.
module.exports = require("./torch")
