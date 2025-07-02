"""
Combine Ukraine Salt Caverns with Existing EU Salt Caverns

This script combines the Ukraine salt cavern shapefile (created with blue threshold detection)
with the existing EU salt cavern shapefile to create a comprehensive European salt cavern dataset.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path


def combine_ukraine_with_eu(ukraine_shp_path, eu_shp_path, output_path, ukraine_boundary_path=None):
    """
    Combine Ukraine salt caverns with existing EU salt caverns
    Removes any existing salt caverns within Ukraine boundaries and replaces with new Ukraine data
    
    Parameters:
    -----------
    ukraine_shp_path : str
        Path to Ukraine salt cavern shapefile
    eu_shp_path : str 
        Path to existing EU salt cavern shapefile (salt_cave/final.shp)
    output_path : str
        Path for combined output shapefile
    ukraine_boundary_path : str, optional
        Path to Ukraine country boundary shapefile. If None, will try to find in resources/
    """
    
    print("=== Combining Ukraine with EU Salt Caverns ===")
    
    # Load Ukraine boundary
    if ukraine_boundary_path is None:
        # Try to find Ukraine boundary in common locations
        possible_paths = [
            "UA_shape_file/world-administrative-boundaries.shp",
            "../resources/country_shapes.geojson",
            "../gas_network_exploring/salt_cave/eu_shapes/country_shapes.geojson",
            "../data/country_shapes.geojson"
        ]
        ukraine_boundary_path = None
        for path in possible_paths:
            if Path(path).exists():
                ukraine_boundary_path = path
                break
        
        if ukraine_boundary_path is None:
            print("❌ Ukraine boundary file not found. Please provide path to country boundaries.")
            print("   Looking for files like country_shapes.geojson or similar...")
            return None
    
    print(f"Loading country boundaries: {ukraine_boundary_path}")
    boundaries_gdf = gpd.read_file(ukraine_boundary_path)
    
    # Find Ukraine boundary (try different possible country codes/names)
    ukraine_boundary = None
    ukraine_identifiers = ['UA', 'UKR', 'Ukraine', 'ukraine', 'UKRAINE', 'UKR ', ' UKR']
    
    for col in boundaries_gdf.columns:
        if col.lower() in ['name', 'country', 'iso', 'iso_a2', 'iso_a3', 'admin', 'iso3', 'country_code']:
            for identifier in ukraine_identifiers:
                ukraine_match = boundaries_gdf[boundaries_gdf[col].astype(str).str.strip() == identifier]
                if not ukraine_match.empty:
                    ukraine_boundary = ukraine_match.iloc[0:1]  # Take first match
                    print(f"✓ Found Ukraine boundary using {col}='{identifier}'")
                    break
            if ukraine_boundary is not None:
                break
    
    if ukraine_boundary is None:
        print("❌ Ukraine boundary not found in the boundary file.")
        print("Available values in boundary file:")
        for col in boundaries_gdf.columns:
            if col.lower() in ['name', 'country', 'iso', 'iso_a2', 'iso_a3', 'admin', 'country_code']:
                unique_vals = boundaries_gdf[col].unique()[:15]  # Show first 15 values
                print(f"  {col}: {unique_vals}...")
        return None
    
    # Load Ukraine salt caverns
    print(f"Loading Ukraine salt caverns: {ukraine_shp_path}")
    ukraine_gdf = gpd.read_file(ukraine_shp_path)
    ukraine_gdf = ukraine_gdf[ukraine_gdf['val_kwhm3'] > 0]  # Remove background
    
    print(f"✓ Loaded {len(ukraine_gdf)} Ukraine salt cavern polygons")
    print(f"  Capacity range: {ukraine_gdf['val_kwhm3'].min():.1f} - {ukraine_gdf['val_kwhm3'].max():.1f} kWh/m³")
    
    # Load existing EU salt caverns
    print(f"Loading EU salt caverns: {eu_shp_path}")
    eu_gdf = gpd.read_file(eu_shp_path)
    eu_gdf = eu_gdf[eu_gdf['val_kwhm3'] > 0]  # Remove background
    
    print(f"✓ Loaded {len(eu_gdf)} EU salt cavern polygons")
    print(f"  Capacity range: {eu_gdf['val_kwhm3'].min():.1f} - {eu_gdf['val_kwhm3'].max():.1f} kWh/m³")
    
    # Ensure Ukraine boundary has same CRS as EU data
    if ukraine_boundary.crs != eu_gdf.crs:
        print(f"Converting Ukraine boundary CRS from {ukraine_boundary.crs} to {eu_gdf.crs}")
        ukraine_boundary = ukraine_boundary.to_crs(eu_gdf.crs)
    
    # Remove any existing salt caverns within Ukraine territory
    print("Removing existing salt caverns within Ukraine territory...")
    original_count = len(eu_gdf)
    
    # Use overlay to find salt caverns outside Ukraine
    eu_outside_ukraine = gpd.overlay(eu_gdf, ukraine_boundary, how='difference')
    
    print(f"✓ Removed {original_count - len(eu_outside_ukraine)} salt cavern polygons from Ukraine territory")
    print(f"✓ Kept {len(eu_outside_ukraine)} EU salt cavern polygons outside Ukraine")
    
    # Ensure both have the same CRS
    if ukraine_gdf.crs != eu_outside_ukraine.crs:
        print(f"Converting Ukraine CRS from {ukraine_gdf.crs} to {eu_outside_ukraine.crs}")
        ukraine_gdf = ukraine_gdf.to_crs(eu_outside_ukraine.crs)
    
    # Clip Ukraine salt caverns to only those within Ukraine boundary
    print("Clipping Ukraine salt caverns to Ukraine boundary...")
    ukraine_clipped = gpd.overlay(ukraine_gdf, ukraine_boundary, how='intersection')
    
    print(f"✓ Clipped Ukraine data from {len(ukraine_gdf)} to {len(ukraine_clipped)} polygons within Ukraine boundary")
    
    # Update ukraine_gdf to use the clipped version
    ukraine_gdf = ukraine_clipped
    
    # Validate Ukraine capacity against expected total (89.8 TWh)
    validation_results = validate_ukraine_capacity(ukraine_gdf, total_twh=89.8)
    
    # Ensure columns match - preserve the original EU structure
    ukraine_cols = set(ukraine_gdf.columns)
    eu_cols = set(eu_outside_ukraine.columns)
    
    print(f"Ukraine columns: {list(ukraine_cols)}")
    print(f"EU columns: {list(eu_cols)}")
    
    # Use the original EU shapefile column structure as the template
    target_columns = list(eu_outside_ukraine.columns)
    
    # Ensure Ukraine data has the same columns as EU data
    for col in target_columns:
        if col not in ukraine_gdf.columns:
            if col == 'geometry':
                continue  # geometry column is already there
            elif col == 'val_kwhm3':
                continue  # capacity column should already exist
            else:
                # Add missing column with default value
                ukraine_gdf[col] = 0 if ukraine_gdf.dtypes.get(col, 'object') in ['int64', 'float64'] else ''
                print(f"Added missing column '{col}' to Ukraine data")
    
    # Ensure both datasets have the same column order and types
    ukraine_subset = ukraine_gdf[target_columns].copy()
    eu_subset = eu_outside_ukraine[target_columns].copy()
    
    # Combine the datasets
    print("\n--- Combining Datasets ---")
    combined_gdf = pd.concat([eu_subset, ukraine_subset], ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, crs=eu_outside_ukraine.crs)
    
    # Fix any invalid geometries
    print("Fixing geometries...")
    combined_gdf['geometry'] = combined_gdf.geometry.buffer(0)
    
    # Calculate statistics
    print("\n--- Combined Dataset Statistics ---")
    print(f"Total polygons: {len(combined_gdf)}")
    print(f"  EU polygons: {len(eu_subset)}")
    print(f"  Ukraine polygons: {len(ukraine_subset)}")
    
    # Calculate total area
    total_area_km2 = combined_gdf.to_crs('EPSG:3395').area.sum() / 1e6
    print(f"Total area: {total_area_km2:.1f} km²")
    
    # Show capacity statistics if available
    if 'val_kwhm3' in combined_gdf.columns:
        capacity_stats = combined_gdf['val_kwhm3'].describe()
        print(f"\nCapacity statistics (kWh/m³):")
        print(f"  Min: {capacity_stats['min']:.1f}")
        print(f"  Max: {capacity_stats['max']:.1f}")
        print(f"  Mean: {capacity_stats['mean']:.1f}")
        print(f"  Count: {capacity_stats['count']:.0f}")
    
    # If country column exists, show stats by country
    if 'country' in combined_gdf.columns:
        country_stats = combined_gdf.groupby('country').size()
        print(f"\nPolygons by country:")
        for country, count in country_stats.items():
            print(f"  {country}: {count}")
        
        if 'val_kwhm3' in combined_gdf.columns:
            capacity_by_country = combined_gdf.groupby('country')['val_kwhm3'].agg(['count', 'min', 'max', 'mean'])
            print("\nCapacity by country:")
            print(capacity_by_country)
        
        # Calculate total area by country
        area_stats = combined_gdf.groupby('country').apply(
            lambda x: x.to_crs('EPSG:3395').area.sum() / 1e6  # Convert to km²
        )
        print(f"\nTotal area by country (km²):")
        for country, area in area_stats.items():
            print(f"  {country}: {area:.1f} km²")
    
    # Save combined dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_gdf.to_file(output_path)
    print(f"\n✅ SUCCESS: Saved combined dataset to {output_path}")
    
    return combined_gdf


def validate_ukraine_capacity(ukraine_gdf, total_twh=89.8):
    """
    Validate Ukraine salt cavern capacity against expected total
    
    Parameters:
    -----------
    ukraine_gdf : GeoDataFrame
        Ukraine salt cavern data
    total_twh : float
        Expected total capacity in TWh (default: 89.8 TWh)
    """
    
    print(f"\n=== Ukraine Capacity Validation ===")
    print(f"Expected total capacity: {total_twh} TWh")
    
    # Calculate total area in different projections for robustness
    ukraine_area_km2 = ukraine_gdf.to_crs('EPSG:3395').area.sum() / 1e6  # World Mercator
    print(f"Detected surface area: {ukraine_area_km2:.2f} km²")
    
    # Check if all Ukraine polygons have the expected capacity value (400 kWh/m³)
    if 'val_kwhm3' in ukraine_gdf.columns:
        unique_capacities = ukraine_gdf['val_kwhm3'].unique()
        print(f"Capacity values found: {unique_capacities}")
        
        if len(unique_capacities) == 1 and unique_capacities[0] == 400:
            print("✓ All Ukraine polygons have correct capacity value (400 kWh/m³)")
            capacity_kwhm3 = 400
        else:
            print("⚠ Warning: Unexpected capacity values found")
            capacity_kwhm3 = ukraine_gdf['val_kwhm3'].mean()
            print(f"Using mean capacity: {capacity_kwhm3:.1f} kWh/m³")
    else:
        print("⚠ Warning: No capacity column found, assuming 400 kWh/m³")
        capacity_kwhm3 = 400
    
    # Validation calculations
    target_wh = total_twh * 1e12  # Convert TWh to Wh
    density_wh_m3 = capacity_kwhm3 * 1000  # Convert kWh/m³ to Wh/m³
    
    # Sanity check: Ukraine total area
    ukraine_total_area_km2 = 603628  # Official Ukraine area
    area_percentage = (ukraine_area_km2 / ukraine_total_area_km2) * 100
    print(f"Salt cavern area is {area_percentage:.3f}% of Ukraine's total area ({ukraine_total_area_km2:,} km²)")
    
    if area_percentage > 10:
        print("⚠ Warning: Detected area seems very large for salt caverns")
    elif area_percentage < 0.001:
        print("⚠ Warning: Detected area seems very small")
    else:
        print("✓ Detected area percentage is reasonable for salt cavern distribution")
    
    # Forward validation: Required area for target capacity
    depth_scenarios = [100, 200, 500]  # Realistic salt cavern depths in meters
    print(f"\nForward validation (required area for {total_twh} TWh at {capacity_kwhm3} kWh/m³):")
    
    for depth_m in depth_scenarios:
        required_volume_m3 = target_wh / density_wh_m3
        required_area_km2 = (required_volume_m3 / depth_m) / 1e6  # Convert m² to km²
        
        print(f"  At {depth_m}m depth: need {required_area_km2:.1f} km² surface area")
        
        if ukraine_area_km2 > 0:
            ratio = required_area_km2 / ukraine_area_km2
            if 0.5 <= ratio <= 2.0:
                status = "✓ Excellent match"
            elif 0.2 <= ratio <= 5.0:
                status = "✓ Good match" 
            elif 0.1 <= ratio <= 10.0:
                status = "⚠ Reasonable"
            else:
                status = "❌ Poor match"
            print(f"    Required/Detected ratio: {ratio:.3f} ({status})")
    
    # Reverse validation: Calculate capacity from detected area
    print(f"\nReverse validation (capacity from {ukraine_area_km2:.1f} km² detected area):")
    for depth_m in depth_scenarios:
        volume_m3 = ukraine_area_km2 * 1e6 * depth_m  # Convert km² to m² then multiply by depth
        capacity_wh = volume_m3 * density_wh_m3
        capacity_twh = capacity_wh / 1e12
        
        ratio = capacity_twh / total_twh
        if 0.5 <= ratio <= 2.0:
            status = "✓ Excellent match"
        elif 0.2 <= ratio <= 5.0:
            status = "✓ Good match"
        elif 0.1 <= ratio <= 10.0:
            status = "⚠ Reasonable"
        else:
            status = "❌ Poor match"
            
        print(f"  At {depth_m}m depth: {capacity_twh:.1f} TWh capacity")
        print(f"    Detected/Target ratio: {ratio:.3f} ({status})")
    
    # Overall assessment
    best_match_found = False
    for depth_m in depth_scenarios:
        volume_m3 = ukraine_area_km2 * 1e6 * depth_m
        capacity_wh = volume_m3 * density_wh_m3
        capacity_twh = capacity_wh / 1e12
        ratio = capacity_twh / total_twh
        if 0.2 <= ratio <= 5.0:
            best_match_found = True
            break
    
    if best_match_found:
        print(f"\n✅ VALIDATION: Ukraine salt cavern detection appears reasonable")
        print(f"   Using {capacity_kwhm3} kWh/m³ density with detected area of {ukraine_area_km2:.1f} km²")
    else:
        print(f"\n⚠️  VALIDATION: Results may need review")
        print(f"   Consider adjusting detection thresholds or capacity assumptions")
    
    return {
        'detected_area_km2': ukraine_area_km2,
        'capacity_kwhm3': capacity_kwhm3,
        'area_percentage': area_percentage,
        'best_match_found': best_match_found
    }


def main():
    """Main function with user input"""
    print("=== Salt Cavern Dataset Combiner ===")
    
    # Default paths
    default_ukraine = "ukraine_results/results/ukraine_salt_caverns.shp"
    default_eu = "../salt_cave/final.shp"
    default_output = "../salt_cave/final_with_ukraine.shp"
    default_boundary = "UA_shape_file/world-administrative-boundaries.shp"  # Ukraine boundary shapefile
    
    ukraine_path = input(f"Ukraine shapefile path (default: {default_ukraine}): ").strip()
    if not ukraine_path:
        ukraine_path = default_ukraine
    
    eu_path = input(f"EU shapefile path (default: {default_eu}): ").strip()
    if not eu_path:
        eu_path = default_eu
    
    output_path = input(f"Output path (default: {default_output}): ").strip()
    if not output_path:
        output_path = default_output
    
    boundary_path = input(f"Ukraine boundary file (press Enter to auto-detect): ").strip()
    if not boundary_path:
        boundary_path = default_boundary
    
    try:
        # Check if files exist
        if not Path(ukraine_path).exists():
            print(f"❌ Ukraine shapefile not found: {ukraine_path}")
            return
        
        if not Path(eu_path).exists():
            print(f"❌ EU shapefile not found: {eu_path}")
            return
        
        result_gdf = combine_ukraine_with_eu(ukraine_path, eu_path, output_path, boundary_path)
        
        if result_gdf is None:
            print("\n❌ Failed to combine datasets")
            return
        
        print(f"\n🎉 Successfully combined datasets!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Total polygons: {len(result_gdf)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
