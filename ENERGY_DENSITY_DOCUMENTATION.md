# Energy Density Documentation

## Salt Cavern Energy Density Values Used in the Project

### EU Salt Caverns (Original Notebook & build_hydrogen_map.py)
- **Method**: Multi-level color classification with K-means clustering
- **Energy Density Levels**: 6 categories
  - 200 kWh/m³ (Dark Purple: RGB [97, 78, 168])
  - 250 kWh/m³ (Teal: RGB [58, 197, 163])
  - 300 kWh/m³ (Light Green: RGB [226, 247, 139])
  - 350 kWh/m³ (Yellow: RGB [255, 224, 122])
  - 400 kWh/m³ (Orange: RGB [255, 100, 50])
  - 450 kWh/m³ (Dark Red: RGB [177, 0, 67])

### Ukraine Salt Caverns (Blue Threshold Approach)
- **Method**: Blue threshold detection for single-color map
- **Energy Density**: **400 kWh/m³** (single value)
- **Total Storage Potential**: **89.8 TWh** (used for validation)
- **Justification**: Ukraine map contains predominantly blue regions representing salt cavern potential, assigned a single capacity value corresponding to the 4th level (orange) in the original EU system
- **Color Detection**: Blue threshold with precise color range: R(55-71), G(59-72), B(192-199), Blue > Red+50, Blue > Green+50
- **Validation Approach**: Detailed capacity validation performed in `combine_with_eu.py` during integration
- **Geographic Scope**: Processes entire image area - geographic filtering handled during combination with EU data

## Alignment Strategy

1. **Consistency**: Both Ukraine and EU datasets use energy density values from the same scale (200-450 kWh/m³)
2. **Ukraine Assignment**: All detected Ukraine salt caverns are assigned 400 kWh/m³, which corresponds to the "orange" level in the original EU legend
3. **Integration**: When combining datasets, Ukraine (400 kWh/m³) will integrate seamlessly with the existing EU multi-level system

## Data Processing Alignment

### Original EU Process (Salt_cave_extraction_final_version.ipynb)

1. Load original salt cave PNG image
2. Apply **variance-based background filtering** (threshold=10)
3. Use **K-means clustering** with 6 predefined RGB colors
4. Create **prediction array** with background=255, clusters=0-5
5. Extract 6 separate layers, each with specific energy density
6. Georeference each layer with GCPs (GDAL + rasterio)
7. Convert each layer to shapefile with `val_kwhm3` column
8. Combine all layers into final shapefile with geometry.buffer(0)

### Ukraine Process (process_ukraine_blue_threshold.py) - Updated to Align

1. Load Ukraine salt cave image  
2. Apply **variance-based background filtering** (threshold=15, same approach)
3. Use **blue pixel detection with stricter criteria** (Blue > Red+20, Blue > Green+20, Blue > 120)
4. Create **prediction array** with background=255, cluster=0 (same structure)
5. Extract single layer with 400 kWh/m³ energy density
6. Georeference layer with Ukraine-specific GCPs (same GDAL + rasterio approach)
7. Convert to shapefile with same column structure (`val_kwhm3`)
8. Apply geometry.buffer(0) fix (same as original)
9. **Note**: Geographic filtering and validation moved to `combine_with_eu.py`

### Build Script Process (build_hydrogen_map.py)

- Maintains the original 6-level EU approach
- Uses identical capacity list: [200, 250, 300, 350, 400, 450] kWh/m³
- Uses identical RGB colors as original notebook
- Preserves the same processing pipeline and output format

## Final Combined Dataset
- **EU regions**: Multi-level energy densities (200-450 kWh/m³)
- **Ukraine regions**: Single energy density (400 kWh/m³)
- **Column consistency**: All regions use `val_kwhm3` column for energy density
- **Geographic consistency**: Both use EPSG:4326 coordinate system
- **Methodological consistency**: Both use GCP-based georeferencing and similar vectorization approaches

## Notes
- The choice of 400 kWh/m³ for Ukraine places it in the upper-middle range of the EU scale
- This value represents a reasonable estimate for Ukraine's salt cavern storage potential
- The single-value approach for Ukraine is appropriate given the single-color nature of the source map
