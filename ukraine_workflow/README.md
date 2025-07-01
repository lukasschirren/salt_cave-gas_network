# Ukraine Salt Cavern Workflow

This directory contains the complete workflow for processing Ukraine salt cavern data and integrating it with the existing EU salt cavern dataset.

## Files in this directory:

### Scripts
- **`create_ukraine_gcps_reliable.py`** - Interactive tool for creating Ground Control Points (GCPs) for georeferencing Ukraine salt cavern images
- **`process_ukraine_blue_threshold.py`** - Main processing script that uses blue threshold detection to extract salt cavern areas from Ukraine images
- **`combine_with_eu.py`** - Script to combine Ukraine salt cavern data with existing EU salt cavern dataset

### Documentation
- **`UKRAINE_PROCESSING_GUIDE.md`** - Detailed step-by-step guide for the entire Ukraine processing workflow

### Output Files
- **`ukraine_salt_caverns.shp`** (+ .dbf, .shx, .prj) - Processed Ukraine salt cavern shapefile

## Quick Start

### Step 1: Create GCPs for your Ukraine image
```bash
python create_ukraine_gcps_reliable.py
```

### Step 2: Process Ukraine salt caverns with blue threshold detection
```bash
python process_ukraine_blue_threshold.py
```

### Step 3: Combine with existing EU dataset
```bash
python combine_with_eu.py
```

## Key Features

### Blue Threshold Detection
- Automatically detects different shades of blue in salt cavern maps
- Handles light gray background (RGB: 226, 226, 226)
- Creates multiple capacity categories based on blue intensity
- All salt caverns marked as "onshore" for Ukraine

### GCP Creation
- Interactive point-and-click interface
- Menu-based system for reliable saving
- Automatic coordinate entry with descriptions
- Supports various image formats

### Dataset Integration
- Combines Ukraine data with existing EU salt cavern shapefile
- Ensures consistent schema and projections
- Provides statistics and validation
- Creates comprehensive European dataset

## Input Requirements
- Ukraine salt cavern image (JPG, PNG, etc.)
- At least 4 Ground Control Points for georeferencing
- Existing EU salt cavern shapefile (../salt_cave/final.shp)

## Output
- Georeferenced Ukraine salt cavern shapefile
- Combined EU + Ukraine salt cavern dataset
- Processing statistics and validation reports

## Notes
- Background color for Ukraine images: RGB(226, 226, 226)
- All Ukraine salt caverns are classified as "onshore"
- Uses EPSG:4326 (WGS84) coordinate system
- Supports blue threshold detection for capacity classification
