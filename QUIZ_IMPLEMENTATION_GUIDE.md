# Quiz Drag & Drop - Datadog Implementation Guide

## Files Created for Your Quiz

1. **`quiz_drag_drop_steps.json`** - Semantic matching (matches questions to correct categories)
2. **`quiz_drag_drop_alternative.json`** - Position-based matching (simpler, more reliable)

## Your Quiz Structure Analysis

Based on your HTML, the quiz has:

### Draggable Options (5 items):
1. "What is the rationale for AI decisions?"
2. "Who is responsible in 'automated decisioning?"
3. "How to manage monopoly over data?"
4. "How to control autonomous weapons?"
5. "E.g. should machines pay taxes?"

### Drop Zones (5 categories):
1. "Transparency and traceability" 
2. "Responsibility and accountability"
3. "Free access to data"
4. "Military use of AI"
5. "Equal treatment of humans & machines"

## Implementation Options

### Option 1: Semantic Matching (Recommended)
Uses the first file (`quiz_drag_drop_steps.json`) - matches questions to their logical categories:

- **Rationale** → **Transparency and traceability**
- **Responsibility** → **Responsibility and accountability** 
- **Monopoly** → **Free access to data**
- **Weapons** → **Military use of AI**
- **Taxes** → **Equal treatment of humans & machines**

### Option 2: Position-Based (Fallback)
Uses the second file (`quiz_drag_drop_alternative.json`) - simply drags first item to first zone, second to second, etc.

## Steps to Add to Your Existing Datadog Test

1. **Copy the steps** from either JSON file
2. **Insert after** your existing navigation steps
3. **Add these steps before** any submission steps

## Example Integration

```json
{
  "steps": [
    // ... your existing login/navigation steps ...
    
    // Insert the quiz drag & drop steps here
    {
      "type": "waitForElement",
      "name": "Wait for Quiz Question Content to Load",
      "params": {
        "element": {
          "multiLocator": {
            "ab": "quiz-question-content",
            "at": "//*[contains(@class, \"quiz__question-content\")]",
            "cl": "//*[contains(@class, \"MuiCardContent-root\")]",
            "co": ".quiz__question-content"
          }
        }
      },
      "timeout": 30
    },
    // ... rest of drag & drop steps ...
    
    // ... your existing submission steps ...
  ]
}
```

## Key Selector Elements from Your HTML

### Draggable Items:
- **Class**: `quiz__matching-option MuiBox-root css-0`
- **Attribute**: `draggable="true"`
- **Container**: `.quiz__matching-options-container`

### Drop Zones:
- **Class**: `quiz__matching-dropzone MuiBox-root css-0`
- **Placeholder**: "Drop answer here"
- **Container**: `.quiz__matching-dropzone-container`

## Verification Steps Included

Both approaches include verification:
- Wait for elements to load
- Confirm drag operations complete
- Verify drop zones are filled

## Troubleshooting Tips

### If Semantic Matching Fails:
- Use the position-based alternative
- Check that question text hasn't changed
- Verify the categories still match

### If Drag & Drop Doesn't Work:
1. **Increase timeouts** - MaterialUI animations may need more time
2. **Add wait steps** between drag operations
3. **Check element visibility** - ensure items are in viewport

### Common Issues:
- **Elements not found**: The MaterialUI classes might have different CSS hashes
- **Timing issues**: Add longer waits between operations
- **Viewport problems**: Elements might need scrolling into view

## Customization Notes

### For Different Questions:
Update the text matching in the semantic version:
```json
"at": "//*[contains(@class, \"quiz__matching-option\") and contains(., \"YOUR_QUESTION_TEXT\")]"
```

### For Different Categories:
Update the category matching:
```json  
"at": "//*[contains(@class, \"quiz__matching-pair\") and contains(., \"YOUR_CATEGORY\")]//*[contains(@class, \"quiz__matching-dropzone\")]"
```

## Next Steps

1. **Test the semantic version first** (more accurate)
2. **If issues occur, switch to position-based** (more reliable)
3. **Adjust timeouts** based on your site's performance
4. **Add to your existing Datadog test** after navigation steps

Both versions will work with your Material-UI quiz structure!