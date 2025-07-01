# Ukraine Salt Cavern Processing Guide

This guide walks you through creating a salt cavern shapefile for Ukraine and combining it with existing EU data.

## Overview

Your situation:
- You have a Ukraine salt cavern image (single color, not multi-color like the EU version)
- You have total capacity data from a paper (in TWh)
- You want to create a shapefile similar to the EU `final.shp`
- You want to combine Ukraine data with existing EU data

## Process Overview

1. **Create Ground Control Points (GCPs)** for your Ukraine image
2. **Process the single-color image** to create salt cavern areas
3. **Generate shapefile** with uniform capacity density
4. **Combine with EU data** (optional)

## Step 1: Setup Environment

```powershell
# Create conda environment
conda env create -f environment_minimal.yml
conda activate ukraine-salt-cavern
```

## Step 2: Create Ground Control Points (GCPs)

You need to identify geographic coordinates for points in your Ukraine image.

### Prepare Reference Points

Before running the GCP creator, gather some known geographic points in Ukraine:

**Major Cities:**
- Kiev (Kyiv): 50.4501°N, 30.5234°E
- Kharkiv: 49.9935°N, 36.2304°E  
- Odessa: 46.4825°N, 30.7233°E
- Lviv: 49.8397°N, 24.0297°E
- Dnipro: 48.4647°N, 35.0462°E

**Border Points:**
- Western border (with Poland): ~50.5°N, 23.5°E
- Eastern border: ~49.5°N, 40.0°E
- Northern border (with Belarus): ~51.5°N, 31.0°E
- Southern border (Black Sea coast): ~45.5°N, 33.0°E

### Run GCP Creator

```powershell
python create_ukraine_gcps.py
```

**Instructions:**
1. Enter path to your Ukraine salt cavern image
2. The image will open in a window
3. Click on recognizable points (cities, borders, coastline)
4. Enter the latitude/longitude for each point
5. Add at least 4 GCPs, ideally spread across the image
6. Press 's' to save when done

**Tips:**
- Use corner points of the image if you know the geographic bounds
- Cities are good reference points if visible
- Coast lines and borders are excellent references
- More GCPs = better accuracy (aim for 6-8 if possible)

## Step 3: Process Ukraine Salt Cavern Image

```powershell
python process_ukraine_salt_caverns.py
```

**You'll be prompted for:**
1. **Image path**: Your Ukraine salt cavern image
2. **GCP path**: The JSON file created in Step 2
3. **Total capacity**: Total TWh from your paper
4. **Color detection**: Auto-detect or manually specify the cavern color
5. **Combination**: Whether to combine with EU data

### Example Input:

```
Enter path to Ukraine salt cavern image: ukraine_salt_caverns.png
Enter path to GCP JSON file: ukraine_salt_caverns_GCP_Points.json
Enter total Ukraine salt cavern capacity (TWh): 150.5
Auto-detect salt cavern color? (y/n): y
```

## Step 4: Understanding the Output

The script creates several files in `ukraine_salt_cavern_output/`:

1. **ukraine_salt_cavern.shp** - Main shapefile with salt cavern polygons
2. **ukraine_salt_cavern.geojson** - Same data in GeoJSON format
3. **ukraine_storage_potential_kwh.csv** - Capacity summary
4. **ukraine_salt_cavern_mask.png** - Binary mask of cavern areas
5. Various intermediate processing files

## Step 5: Combine with EU Data (Optional)

If you want to create a combined EU+Ukraine dataset:

```powershell
# When prompted by the script:
Combine with existing EU salt cavern data? (y/n): y
Enter path to EU salt cavern shapefile: salt_cave/final.shp  
Enter path to EU capacity CSV: salt_cave/storage_potential_eu_kwh.csv
```

This creates:
- **combined_salt_cavern_with_ukraine.shp** - Combined shapefile
- **combined_storage_potential_kwh.csv** - Combined capacity data

## Troubleshooting

### GCP Issues
- **Image won't load**: Check file path and format (PNG, JPG, TIF supported)
- **Can't find reference points**: Use online maps to identify coordinates
- **Poor georeferencing**: Add more GCPs, especially at corners

### Color Detection Issues
- **No caverns detected**: Try manual color specification
- **Too much area detected**: Reduce color tolerance
- **Too little area detected**: Increase color tolerance or check color values

### Capacity Issues
- **Unrealistic density**: Check total TWh input value
- **No output polygons**: Check that mask contains cavern areas

## Manual Color Specification

If auto-detection doesn't work, you can manually specify colors:

1. Open your image in an image editor
2. Use color picker on salt cavern areas  
3. Note the RGB values
4. Convert to BGR (reverse the order) for OpenCV
5. Enter these values when prompted

**Example:**
- RGB: (120, 180, 90) → BGR: (90, 180, 120)

## File Structure After Processing

```
ukraine_salt_cavern_output/
├── ukraine_salt_cavern.shp          # Main output shapefile
├── ukraine_salt_cavern.geojson      # GeoJSON version
├── ukraine_storage_potential_kwh.csv # Capacity summary
├── ukraine_salt_cavern_mask.png     # Processing mask
├── ukraine_salt_cavern_georef.tif   # Intermediate georeferenced raster
└── ukraine_salt_cavern_warped.tif   # Final warped raster

combined_output/ (if combining)
├── combined_salt_cavern_with_ukraine.shp
└── combined_storage_potential_kwh.csv
```

## Next Steps

Once you have the Ukraine shapefile:

1. **Validate the output** by opening in QGIS or similar GIS software
2. **Check coordinates** ensure caverns are in the right locations
3. **Verify capacity values** compare with your source paper
4. **Use in PyPSA-Eur** by replacing or combining with existing salt cavern data

## Quality Control Checklist

- [ ] GCPs cover the full extent of the image
- [ ] At least 4 GCPs with good geographic distribution  
- [ ] Salt cavern mask covers expected areas
- [ ] Output shapefile opens correctly in GIS software
- [ ] Polygons are in correct geographic locations
- [ ] Total capacity matches your input value
- [ ] Capacity density is reasonable (compare with EU values)

## Common Issues and Solutions

**Issue**: Shapefile has no features
**Solution**: Check that the mask detected cavern areas, adjust color tolerance

**Issue**: Polygons in wrong location  
**Solution**: Verify GCP coordinates, add more reference points

**Issue**: Capacity density too high/low
**Solution**: Double-check total TWh input, verify mask area calculation

**Issue**: Can't combine with EU data
**Solution**: Ensure EU files exist and have compatible column structure
