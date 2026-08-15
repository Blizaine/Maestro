// Maestro's managed models are public and require no account. This explicit
// action is only for users who want custom gated assets or higher Hugging Face
// download limits. Force a fresh flow so it can also replace a stale token.
module.exports = {
  run: [{
    method: "hf.login",
    params: {
      force: true,
      wait: true
    }
  }]
}
