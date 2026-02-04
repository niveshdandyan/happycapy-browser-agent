#!/bin/bash
# Example: Google Search with browser-use
# Performs a search and captures results

echo "🔍 Google Search Example"
echo "========================"

# Open Google
echo ""
echo "Opening Google..."
browser-use open https://google.com
sleep 1

# Get page state to find search box
echo ""
echo "Finding search box..."
browser-use state

# The search box is usually one of the first textarea/input elements
# Common indices: 0-10 range, look for "textarea" or input with "Search"

# Type search query (adjust index based on state output)
# Typically the search textarea is around index 4-8
echo ""
echo "Entering search query..."
# Try the common search box selector via JavaScript first
browser-use eval "document.querySelector('textarea[name=\"q\"]').value = 'browser-use AI automation'"
browser-use keys "Enter"

# Wait for results
echo ""
echo "Waiting for results..."
browser-use wait text "results"
sleep 2

# Capture results
echo ""
echo "Capturing search results..."
browser-use screenshot search_results.png

# Get page title
echo ""
echo "Page info:"
browser-use get title

# Extract result titles via JavaScript
echo ""
echo "Extracting top results..."
browser-use eval "
Array.from(document.querySelectorAll('h3')).slice(0, 5).map(h => h.textContent)
"

# Scroll for more results
echo ""
echo "Scrolling for more results..."
browser-use scroll down
browser-use scroll down
browser-use screenshot search_results_scrolled.png

# Close browser
echo ""
echo "Cleaning up..."
browser-use close

echo ""
echo "✅ Search complete!"
echo "Screenshots saved: search_results.png, search_results_scrolled.png"
