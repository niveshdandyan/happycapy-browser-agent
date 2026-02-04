#!/bin/bash
# Example: Form Automation with browser-use
# Fills out a contact form automatically

echo "📝 Form Automation Example"
echo "=========================="

# Navigate to form page
echo "Opening form page..."
browser-use open https://www.w3schools.com/html/html_forms.asp

# Wait for page load
sleep 2

# Get current state to see elements
echo ""
echo "Analyzing form elements..."
browser-use state

# Scroll to find the form
echo ""
echo "Scrolling to form..."
browser-use scroll down
browser-use scroll down

# Get state again after scrolling
echo ""
echo "Form elements after scroll:"
browser-use state

# Screenshot before filling
echo ""
echo "Taking before screenshot..."
browser-use screenshot form_before.png

# Example: If form has these indices from state:
# [10] input "First name"
# [11] input "Last name"
# [12] button "Submit"

# Uncomment and adjust indices based on browser-use state output:
# browser-use input 10 "John"
# browser-use input 11 "Doe"
# browser-use screenshot form_filled.png
# browser-use click 12

# Close browser
echo ""
echo "Cleaning up..."
browser-use close

echo ""
echo "✅ Form example complete!"
echo "Check form_before.png for screenshot"
echo ""
echo "To fill a real form:"
echo "1. Run: browser-use state"
echo "2. Find element indices"
echo "3. Run: browser-use input <index> \"text\""
echo "4. Run: browser-use click <submit-index>"
