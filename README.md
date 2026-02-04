# HappyCapy Browser Agent Skill 🌐

This repository contains the **Browser-Use Skill** for HappyCapy.ai, enabling AI agents to control a real web browser for automation, scraping, and testing.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   ./setup.sh
   # Sets up a virtual environment and installs browser-use
   ```

2. **Activate Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Run an Example**
   ```bash
   ./examples/google_search.sh
   ```

## 📂 Repository Structure

- `setup.sh`: Installation script (Auto-creates `.venv` and installs Chromium)
- `examples/`: Ready-to-run scripts
  - `google_search.sh`: Search Google and extract results
  - `scraping.sh`: E-commerce data extraction example
  - `form_filling.sh`: Automated form submission
  - `screenshots.sh`: Capture screenshots of websites

## 🛠️ Usage with AI Agents

This repository is designed to be used as a **Skill** for Claude Code or other AI agents.

**Core Commands:**
- `browser-use open <url>`
- `browser-use state` (Get clickable elements)
- `browser-use click <index>`
- `browser-use type "text"`
- `browser-use screenshot`

---
*Powered by [browser-use](https://github.com/browser-use/browser-use)*
