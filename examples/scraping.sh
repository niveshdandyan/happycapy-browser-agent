#!/bin/bash
# Example: Web Scraping with browser-use
# Scrapes product data from an e-commerce page

echo "🛒 E-Commerce Scraper Example"
echo "=============================="

# Navigate to products page
echo "Opening product page..."
browser-use open https://webscraper.io/test-sites/e-commerce/allinone

# Wait for page to load
sleep 2

# Get page state
echo ""
echo "Page elements:"
browser-use state

# Scroll to load more products
echo ""
echo "Scrolling to load more..."
browser-use scroll down
browser-use scroll down
browser-use scroll down

# Take screenshot
echo ""
echo "Capturing screenshot..."
browser-use screenshot products.png

# Extract product data via JavaScript
echo ""
echo "Extracting product data..."
browser-use eval "
Array.from(document.querySelectorAll('.thumbnail')).map(el => ({
    title: el.querySelector('.title')?.textContent?.trim(),
    price: el.querySelector('.price')?.textContent?.trim(),
    rating: el.querySelector('.ratings')?.textContent?.trim()
}))
"

# Close browser
echo ""
echo "Cleaning up..."
browser-use close

echo ""
echo "✅ Scraping complete! Check products.png"
