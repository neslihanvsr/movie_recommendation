# Datadog Synthetic Test - Quiz Platform with Drag & Drop

This guide provides the complete setup for a Datadog synthetic test that handles login, navigation to quiz pages, and drag-and-drop question interactions.

## Files Created

1. **`datadog_quiz_test.json`** - Complete Datadog synthetic test configuration
2. **`drag_drop_steps.json`** - Simplified drag & drop steps to add to existing tests

## Test Flow Overview

### 1. Authentication Steps
- Navigate to quiz platform
- Click login button
- Enter username and password
- Submit login form
- Wait for dashboard to load

### 2. Navigation Steps
- Navigate to quiz section
- Wait for quiz page to load
- Scroll past video content to reach quiz questions

### 3. Drag & Drop Interaction Steps
- Wait for drag-and-drop question elements to load
- Perform multiple drag-and-drop operations
- Submit quiz answers
- Verify completion

## Key Features

### Multi-Locator Strategy
Each element uses multiple locator strategies for reliability:
- **ab**: Accessibility-based locators
- **at**: XPath selectors
- **cl**: Class-based XPath
- **co**: CSS selectors

### Drag & Drop Implementation
```json
{
  "type": "dragAndDrop",
  "name": "Drag Answer to Drop Zone",
  "params": {
    "source": {
      "multiLocator": {
        "ab": "draggable-item-1",
        "at": "//*[contains(@class, \"draggable-item\")][1]",
        "cl": "//*[@data-draggable=\"item-1\"]",
        "co": ".draggable-item:first-child"
      }
    },
    "target": {
      "multiLocator": {
        "ab": "drop-zone-1",
        "at": "//*[contains(@class, \"drop-zone\")][1]",
        "cl": "//*[@data-drop-zone=\"zone-1\"]",
        "co": ".drop-zone:first-child"
      }
    }
  },
  "timeout": 30
}
```

## Customization Required

### 1. Update URLs
Replace `https://your-quiz-platform.com` with your actual platform URL.

### 2. Customize Element Selectors
Update the multiLocator values to match your application's HTML structure:

**Login Elements:**
- Login button: Update selectors for your login button
- Username field: Adjust for your username input field
- Password field: Adjust for your password input field

**Navigation Elements:**
- Quiz navigation: Update to match your quiz menu/navigation
- Quiz container: Adjust for your quiz page container

**Drag & Drop Elements:**
- Draggable items: Update class names and data attributes
- Drop zones: Update to match your drop zone selectors
- Question container: Adjust for your question layout

### 3. Configure Variables
Set up your Datadog variables:
```json
"configVariables": [
  {
    "name": "USERNAME",
    "type": "text",
    "example": "your-actual-username"
  },
  {
    "name": "PASSWORD",
    "type": "text",
    "example": "your-actual-password",
    "secure": true
  }
]
```

### 4. Adjust Timeouts
Modify timeout values based on your application's performance:
- Page loads: 60 seconds (adjust for slower pages)
- Element interactions: 30 seconds
- Form submissions: 30 seconds

## Implementation Steps

### Option 1: Use Complete Test
1. Upload `datadog_quiz_test.json` to Datadog
2. Customize URLs and selectors
3. Configure username/password variables
4. Test and adjust as needed

### Option 2: Add to Existing Test
1. Copy steps from `drag_drop_steps.json`
2. Insert after your login and navigation steps
3. Customize selectors for your application
4. Add submit and verification steps

## Video Handling

The test includes a scroll step that bypasses video content:
```json
{
  "type": "scroll",
  "name": "Scroll to Quiz Questions (Skip Video)",
  "params": {
    "element": {
      "multiLocator": {
        "ab": "quiz-questions",
        "at": "//*[contains(@class, \"quiz-questions\") or contains(@class, \"question-container\")]",
        "cl": "//*[@id=\"questions-section\"]",
        "co": "[data-testid=\"quiz-questions\"]"
      }
    }
  },
  "timeout": 30
}
```

This ensures the test focuses on the quiz interactions rather than video playback.

## Troubleshooting

### Common Issues:
1. **Element not found**: Update selectors to match your HTML
2. **Timeout errors**: Increase timeout values for slower loading
3. **Drag & drop failures**: Ensure elements are properly loaded before interaction
4. **Login failures**: Verify username/password variables are correctly set

### Debugging Tips:
1. Use browser developer tools to inspect element selectors
2. Test individual steps in Datadog's synthetic test recorder
3. Add additional wait steps if timing issues occur
4. Use screenshot assertions to verify page state

## Best Practices

1. **Use data attributes** for more reliable element selection
2. **Add wait steps** before critical interactions
3. **Include verification steps** to confirm successful actions
4. **Set appropriate timeouts** based on application performance
5. **Use descriptive step names** for easier debugging

## Monitoring Configuration

- **Frequency**: Every 15 minutes (900 seconds)
- **Locations**: AWS US-East-1 (adjust as needed)
- **Retry policy**: 2 retries with 5-minute intervals
- **Alert priority**: Medium (3)

Adjust these settings based on your monitoring requirements and application criticality.