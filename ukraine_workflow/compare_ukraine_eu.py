"""
Quick comparison between Ukraine salt caverns and EU salt caverns from final.shp
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

def compare_ukraine_eu_salt_caverns():
    """Compare Ukraine and EU salt caverns"""
    
    # Paths
    ukraine_path = Path("ukraine_results/results/ukraine_salt_caverns.shp")
    eu_path = Path("../salt_cave/final.shp")
    
    print("=== Comparing Ukraine vs EU Salt Caverns ===\n")
    
    # Check if files exist
    if not ukraine_path.exists():
        print(f"❌ Ukraine file not found: {ukraine_path}")
        print("   Run the Ukraine processing script first")
        return
    
    if not eu_path.exists():
        print(f"❌ EU file not found: {eu_path}")
        print("   Check the path to the EU salt cavern shapefile")
        return
    
    # Load shapefiles
    print("Loading shapefiles...")
    ukraine_gdf = gpd.read_file(ukraine_path)
    eu_gdf = gpd.read_file(eu_path)
    
    print(f"✓ Loaded Ukraine shapefile: {len(ukraine_gdf)} features")
    print(f"✓ Loaded EU shapefile: {len(eu_gdf)} features")
    
    # Basic statistics
    print("\n--- Basic Statistics ---")
    
    # Ukraine stats
    print("UKRAINE:")
    print(f"  Total features: {len(ukraine_gdf)}")
    if 'val_kwhm3' in ukraine_gdf.columns:
        ukraine_capacities = ukraine_gdf['val_kwhm3'].value_counts().sort_index()
        print("  Capacity distribution:")
        for capacity, count in ukraine_capacities.items():
            print(f"    {capacity} kWh/m³: {count} features")
    
    # Calculate Ukraine area
    ukraine_area_km2 = ukraine_gdf.to_crs('EPSG:3395').area.sum() / 1e6
    print(f"  Total area: {ukraine_area_km2:.2f} km²")
    
    # EU stats
    print("\nEU:")
    print(f"  Total features: {len(eu_gdf)}")
    if 'val_kwhm3' in eu_gdf.columns:
        eu_capacities = eu_gdf['val_kwhm3'].value_counts().sort_index()
        print("  Capacity distribution:")
        for capacity, count in eu_capacities.items():
            print(f"    {capacity} kWh/m³: {count} features")
    
    # Calculate EU area
    eu_area_km2 = eu_gdf.to_crs('EPSG:3395').area.sum() / 1e6
    print(f"  Total area: {eu_area_km2:.2f} km²")
    
    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Ukraine area / EU area ratio: {ukraine_area_km2 / eu_area_km2:.4f}")
    print(f"Ukraine features / EU features ratio: {len(ukraine_gdf) / len(eu_gdf):.4f}")
    
    # Capacity analysis
    if 'val_kwhm3' in ukraine_gdf.columns and 'val_kwhm3' in eu_gdf.columns:
        print(f"\n--- Capacity Analysis ---")
        
        # Ukraine capacity ranges
        ukraine_min = ukraine_gdf['val_kwhm3'].min()
        ukraine_max = ukraine_gdf['val_kwhm3'].max()
        ukraine_mean = ukraine_gdf['val_kwhm3'].mean()
        
        # EU capacity ranges
        eu_min = eu_gdf['val_kwhm3'].min()
        eu_max = eu_gdf['val_kwhm3'].max()
        eu_mean = eu_gdf['val_kwhm3'].mean()
        
        print("Ukraine capacity:")
        print(f"  Min: {ukraine_min} kWh/m³")
        print(f"  Max: {ukraine_max} kWh/m³") 
        print(f"  Mean: {ukraine_mean:.1f} kWh/m³")
        
        print("EU capacity:")
        print(f"  Min: {eu_min} kWh/m³")
        print(f"  Max: {eu_max} kWh/m³")
        print(f"  Mean: {eu_mean:.1f} kWh/m³")
    
    # Geographic bounds comparison
    print(f"\n--- Geographic Bounds ---")
    ukraine_bounds = ukraine_gdf.total_bounds
    eu_bounds = eu_gdf.total_bounds
    
    print("Ukraine bounds (lon_min, lat_min, lon_max, lat_max):")
    print(f"  {ukraine_bounds[0]:.3f}, {ukraine_bounds[1]:.3f}, {ukraine_bounds[2]:.3f}, {ukraine_bounds[3]:.3f}")
    
    print("EU bounds:")
    print(f"  {eu_bounds[0]:.3f}, {eu_bounds[1]:.3f}, {eu_bounds[2]:.3f}, {eu_bounds[3]:.3f}")
    
    # Check for overlap
    ukraine_overlap = (
        ukraine_bounds[0] < eu_bounds[2] and ukraine_bounds[2] > eu_bounds[0] and
        ukraine_bounds[1] < eu_bounds[3] and ukraine_bounds[3] > eu_bounds[1]
    )
    
    if ukraine_overlap:
        print("✓ Geographic bounds overlap - good for integration")
    else:
        print("⚠ No geographic overlap - check coordinate systems")
    
    print(f"\n=== Comparison Complete ===")

if __name__ == "__main__":
    compare_ukraine_eu_salt_caverns()
