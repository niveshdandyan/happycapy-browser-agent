#!/bin/bash
# Example: Screenshot Capture with browser-use
# Takes screenshots of various pages for documentation

echo "📸 Screenshot Capture Example"
echo "=============================="

# Create output directory
mkdir -p screenshots

# Capture Google homepage
echo ""
echo "Capturing Google..."
browser-use open https://google.com
sleep 1
browser-use screenshot screenshots/google.png
echo "✓ Saved screenshots/google.png"

# Capture GitHub
echo ""
echo "Capturing GitHub..."
browser-use open https://github.com
sleep 2
browser-use screenshot screenshots/github.png
echo "✓ Saved screenshots/github.png"

# Capture full page screenshot
echo ""
echo "Capturing full page screenshot..."
browser-use open https://example.com
browser-use screenshot --full screenshots/example_full.png
echo "✓ Saved screenshots/example_full.png"

# Capture with visible browser (for debugging)
echo ""
echo "Capturing with visible browser..."
browser-use close
browser-use --headed open https://wikipedia.org
sleep 3
browser-use screenshot screenshots/wikipedia.png
echo "✓ Saved screenshots/wikipedia.png"

# Close browser
echo ""
echo "Cleaning up..."
browser-use close

echo ""
echo "✅ All screenshots saved to ./screenshots/"
ls -la screenshots/
