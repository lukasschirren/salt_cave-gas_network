"""
Ukraine Salt Cavern Processing with Blue Threshold Detection

This script processes a Ukraine salt cavern image using blue threshold detection
to capture different shades of blue representing salt cavern potential.
Based on the extract_with_threshold approach from the original notebook.
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal, osr
import geopandas as gpd
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry.multipolygon import MultiPolygon
import fiona


def load_ukraine_gcps(gcp_file_path):
    """Load GCPs from JSON file and convert to GDAL format"""
    with open(gcp_file_path, 'r') as f:
        gcp_data = json.load(f)
    
    # Convert to GDAL GCP objects
    object_gcps = []
    for gcp in gcp_data['gcps']:
        # GDAL GCP format: (longitude, latitude, elevation, pixel_x, pixel_y)
        object_gcps.append(gdal.GCP(
            gcp['longitude'], gcp['latitude'], 0, 
            gcp['image_x'], gcp['image_y']
        ))
    
    print(f"Loaded {len(object_gcps)} GCPs from {gcp_file_path}")
    return object_gcps


def extract_blue_threshold(image_path, threshold=10, 
                          blue_colors=[[100, 150, 255]],  # Single blue color for detection
                          capacities=[400],  # Single capacity value: 400 kWh/m³
                          background_color=[226, 226, 226],
                          total_twh=None,
                          ukraine_gcps=None):
    """
    Extract blue salt cavern areas using threshold detection with single capacity value
    
    Parameters:
    -----------
    image_path : str
        Path to Ukraine salt cavern image
    threshold : float
        Variance threshold to filter background pixels
    blue_colors : list
        RGB values for blue detection (single color for all salt caverns)
    capacities : list
        Single capacity value (400 kWh/m³) for all detected salt caverns
    background_color : list
        RGB value of background color [226, 226, 226]
    total_twh : float
        Total TWh capacity for Ukraine (if provided, will calculate density)
    ukraine_gcps : list
        Ground Control Points for creating Ukraine boundary mask
    
    Returns:
    --------
    dict: Dictionary with processed layers and metadata
    """
    
    # Load and prepare image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {img.shape}")
    
    # Flatten image to list and create DataFrame
    img_list = img.reshape((-1, 3))
    img_df = pd.DataFrame(img_list, columns=['R', 'G', 'B'])
    
    # Create Ukraine boundary mask to limit detection to Ukraine territory only
    # Boundary masking disabled - works better without it, combination script handles geographic accuracy
    print("⚠️  Ukraine boundary masking disabled - processing entire image")
    ukraine_mask = np.ones(len(img_df), dtype=bool)
    print(f"Processing entire image: {len(ukraine_mask):,} pixels")
    
    # Calculate variance for each pixel to identify background
    img_df['variance'] = img_df.var(axis=1)
    
    # Identify background pixels (low variance or specific background color)
    background_mask = (
        (img_df['variance'] <= threshold) |
        ((abs(img_df['R'] - background_color[0]) <= 5) &
         (abs(img_df['G'] - background_color[1]) <= 5) &
         (abs(img_df['B'] - background_color[2]) <= 5))
    )
    
    print(f"Background pixels: {background_mask.sum()} ({background_mask.sum()/len(img_df)*100:.1f}%)")
    print(f"Non-background pixels: {(~background_mask).sum()} ({(~background_mask).sum()/len(img_df)*100:.1f}%)")
    
    # Focus on blue pixels only (targeting blue range: RGB(71,72,199) to RGB(55,59,192)) AND within Ukraine
    blue_r_min, blue_r_max = 30, 100    # Red channel range
    blue_g_min, blue_g_max = 30, 110    # Green channel range  
    blue_b_min, blue_b_max = 120, 230  # Blue channel range
    blue_mask = (
        ukraine_mask &                      # ONLY within Ukraine boundaries
        (~background_mask) &
        (img_df['R'] >= blue_r_min) &       # Red channel within range (55-71)
        (img_df['R'] <= blue_r_max) &
        (img_df['G'] >= blue_g_min) &       # Green channel within range (59-72)
        (img_df['G'] <= blue_g_max) &
        (img_df['B'] >= blue_b_min) &       # Blue channel within range (192-199)
        (img_df['B'] <= blue_b_max) &
        (img_df['B'] > img_df['R'] + 50) &  # Blue strongly dominant over red
        (img_df['B'] > img_df['G'] + 50)    # Blue strongly dominant over green
    )
    
    print(f"Blue pixels: {blue_mask.sum()} ({blue_mask.sum()/len(img_df)*100:.1f}%)")
    
    if blue_mask.sum() == 0:
        print("Warning: No blue pixels found!")
        return None
    
    # Cluster blue pixels into different categories using distance-based assignment
    n_clusters = len(blue_colors)
    blue_colors_array = np.array(blue_colors, dtype='float64')
    
    # Create prediction array
    img_df['predict'] = 255  # Background label
    
    # Assign blue pixels to closest predefined blue color
    blue_pixels = img_df[blue_mask][['R', 'G', 'B']].values
    if len(blue_pixels) > 0:
        # Calculate distances to each predefined blue color
        predictions = []
        for pixel in blue_pixels:
            distances = np.sqrt(np.sum((blue_colors_array - pixel) ** 2, axis=1))
            closest_cluster = np.argmin(distances)
            predictions.append(closest_cluster)
        
        img_df.loc[blue_mask, 'predict'] = predictions
    
    # If total TWh is provided, show basic statistics
    if total_twh is not None:
        total_blue_pixels = blue_mask.sum()
        print(f"Total blue pixels detected: {total_blue_pixels:,}")
        print(f"Expected total capacity: {total_twh} TWh")
        print(f"(Detailed validation will be performed in combine_with_eu.py)")
    
    # Generate processed layers
    layers = {}
    for i, capacity in enumerate(capacities):
        # Create binary layer for this capacity level
        layer_data = img_df.copy()
        
        # Set all non-matching pixels to white (255)
        layer_data.loc[layer_data['predict'] != i, ['R', 'G', 'B']] = 255
        
        # Set matching pixels to black (0) for vectorization
        layer_data.loc[layer_data['predict'] == i, ['R', 'G', 'B']] = 0
        
        # Reshape back to image
        layer_img = layer_data[['R', 'G', 'B']].values.reshape(img.shape).astype('uint8')
        
        layers[f'capacity_{capacity}'] = {
            'image': layer_img,
            'capacity': capacity,
            'pixel_count': (layer_data['predict'] == i).sum()
        }
        
        print(f"Layer {i}: {capacity} kWh/m³, {layers[f'capacity_{capacity}']['pixel_count']} pixels")
    
    return {
        'layers': layers,
        'original_shape': img.shape,
        'total_pixels': len(img_df),
        'blue_pixels': blue_mask.sum(),
        'background_pixels': background_mask.sum()
    }


def save_layer_image(layer_img, output_path):
    """Save layer image as grayscale PNG"""
    img_gray = cv2.cvtColor(layer_img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(output_path), img_gray)


def warp_with_gcps(input_path, output_path, gcps, gcp_epsg=4326, output_epsg=4326):
    """
    Georeference image using Ground Control Points
    """
    # Open source dataset and add GCPs
    src_ds = gdal.OpenShared(str(input_path), gdal.GA_ReadOnly)
    gcp_srs = osr.SpatialReference()
    gcp_srs.ImportFromEPSG(gcp_epsg)
    gcp_crs_wkt = gcp_srs.ExportToWkt()
    src_ds.SetGCPs(gcps, gcp_crs_wkt)

    # Define target spatial reference system
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0
    resampling = gdal.GRA_Bilinear

    # Get target dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds, None, dst_wkt, resampling, error_threshold)
    dst_xsize = tmp_ds.RasterXSize
    dst_ysize = tmp_ds.RasterYSize
    dst_gt = tmp_ds.GetGeoTransform()
    tmp_ds = None

    # Create target dataset
    dst_path = str(Path(output_path).with_suffix(".tif"))
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_path, dst_xsize, dst_ysize, src_ds.RasterCount)
    dst_ds.SetProjection(dst_wkt)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.GetRasterBand(1).SetNoDataValue(255)

    # Reproject
    gdal.ReprojectImage(src_ds, dst_ds, None, None, resampling, 0, error_threshold, None, None)
    dst_ds = None

    # Clean up pixel values to binary (0 or 255)
    with rasterio.open(output_path) as dataset:
        kwds = dataset.profile
        band = (dataset.read(1) > 20) * 255

    with rasterio.open(output_path, 'w', **kwds) as dataset:
        dataset.write_band(1, band.astype('uint8'))


def raster2shp(input_path, output_path, capacity, background=255):
    """
    Convert raster to shapefile with capacity values
    """
    with rasterio.open(input_path) as src:
        crs = src.crs
        src_band = src.read(1)
        unique_values = np.unique(src_band)
        shapes = list(rasterio.features.shapes(src_band, transform=src.transform))

    # Shapefile schema
    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'val_kwhm3': 'int', 'country': 'str', 'type': 'str'}
    }

    with fiona.open(output_path, 'w', 'ESRI Shapefile', shp_schema, crs) as shp:
        for pixel_value in unique_values:
            polygons = [shape(geom) for geom, value in shapes if value == pixel_value]
            multipolygon = MultiPolygon(polygons)

            # Set capacity value
            if pixel_value == background:
                pixel_value = 0
            else:
                pixel_value = capacity

            # Write to shapefile
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {
                    'val_kwhm3': int(pixel_value),
                    'country': 'Ukraine',
                    'type': 'onshore'
                }
            })


def create_ukraine_boundary_mask(img_shape, gcps, ukraine_boundary_path=None):
    """
    Create a mask for Ukraine's boundaries to limit salt cavern detection to Ukraine territory only
    """
    
    if ukraine_boundary_path is None:
        # Use default Ukraine boundary file
        ukraine_boundary_path = Path(__file__).parent / "UA_shape_file" / "world-administrative-boundaries.shp"
    
    if not Path(ukraine_boundary_path).exists():
        print(f"⚠️  Ukraine boundary file not found: {ukraine_boundary_path}")
        print("   Proceeding without boundary mask (will process entire image)")
        return np.ones((img_shape[0], img_shape[1]), dtype=bool)
    
    try:
        # Load Ukraine boundary
        ukraine_gdf = gpd.read_file(ukraine_boundary_path)
        
        # Find Ukraine
        country_columns = ['NAME', 'COUNTRY', 'name', 'country', 'NAME_EN', 'ADMIN']
        ukraine_filter = None
        
        for col in country_columns:
            if col in ukraine_gdf.columns:
                ukraine_mask_col = ukraine_gdf[col].str.contains('Ukraine', case=False, na=False)
                if ukraine_mask_col.any():
                    ukraine_filter = ukraine_mask_col
                    print(f"✓ Found Ukraine using column '{col}'")
                    break
        
        if ukraine_filter is not None:
            ukraine_gdf = ukraine_gdf[ukraine_filter]
        else:
            print("⚠️  Could not identify Ukraine in boundary file, using first geometry")
            ukraine_gdf = ukraine_gdf.head(1)
        
        if ukraine_gdf.empty:
            print("⚠️  No Ukraine boundary found in shapefile")
            return np.ones((img_shape[0], img_shape[1]), dtype=bool)
        
        print(f"✓ Loaded Ukraine boundary")
        
        # Calculate image bounds from GCPs
        gcp_lons = [gcp.GCPX for gcp in gcps]
        gcp_lats = [gcp.GCPY for gcp in gcps]
        
        min_lon, max_lon = min(gcp_lons), max(gcp_lons)
        min_lat, max_lat = min(gcp_lats), max(gcp_lats)
        
        # Add some padding
        lon_pad = (max_lon - min_lon) * 0.1
        lat_pad = (max_lat - min_lat) * 0.1
        
        bounds = (min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad)
        
        # Create transform for rasterizing
        img_height, img_width = img_shape[:2]
        from rasterio.transform import from_bounds
        transform = from_bounds(*bounds, img_width, img_height)
        
        # Rasterize Ukraine boundary to create mask
        from rasterio.features import rasterize
        ukraine_geometries = ukraine_gdf.geometry.tolist()
        mask = rasterize(
            ukraine_geometries,
            out_shape=(img_height, img_width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8'
        )
        
        mask_percentage = (mask.sum() / mask.size) * 100
        print(f"✓ Created Ukraine boundary mask: {mask_percentage:.1f}% of image is within Ukraine")
        
        return mask.astype(bool)
        
    except Exception as e:
        print(f"⚠️  Error creating Ukraine boundary mask: {e}")
        print("   Proceeding without boundary mask")
        return np.ones((img_shape[0], img_shape[1]), dtype=bool)


def process_ukraine_salt_caverns(image_path, gcp_file_path, output_dir, total_twh=None):
    """
    Main processing function for Ukraine salt caverns
    
    Parameters:
    -----------
    image_path : str
        Path to Ukraine salt cavern image
    gcp_file_path : str
        Path to GCP JSON file
    output_dir : str
        Output directory for results
    total_twh : float, optional
        Total TWh capacity for Ukraine (89.8 TWh) - used for validation only,
        does not override the fixed 400 kWh/m³ density
    """
    
    # Setup paths
    image_path = Path(image_path)
    gcp_file_path = Path(gcp_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    raster_dir = output_dir / 'temp_raster'
    vector_dir = output_dir / 'temp_vector'
    results_dir = output_dir / 'results'
    
    raster_dir.mkdir(exist_ok=True)
    vector_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    print("=== Processing Ukraine Salt Caverns with Blue Threshold ===")
    print(f"Image: {image_path}")
    print(f"GCPs: {gcp_file_path}")
    print(f"Output: {output_dir}")
    
    # Load GCPs
    gcps = load_ukraine_gcps(gcp_file_path)
    
    # Extract blue threshold layers
    print("\n--- Extracting Blue Threshold Layers (Single Capacity: 400 kWh/m³) ---")
    result = extract_blue_threshold(
        image_path, 
        threshold=15,  # Adjusted for Ukraine image
        blue_colors=[[100, 150, 255]],  # Single blue color for all salt caverns
        capacities=[400],  # Single capacity: 400 kWh/m³
        background_color=[226, 226, 226],
        total_twh=total_twh,
        ukraine_gcps=gcps  # Pass GCPs for boundary mask creation
    )
    
    if result is None:
        print("❌ Failed to extract blue layers")
        return
    
    # Process each layer
    print("\n--- Processing Individual Layers ---")
    gdf_list = []
    
    for layer_name, layer_data in result['layers'].items():
        if layer_data['pixel_count'] == 0:
            print(f"Skipping {layer_name} - no pixels")
            continue
            
        capacity = layer_data['capacity']
        print(f"\nProcessing {layer_name} (capacity: {capacity:.1f} kWh/m³)")
        
        # Save layer image
        png_path = raster_dir / f'{layer_name}.png'
        save_layer_image(layer_data['image'], png_path)
        
        # Georeference with GCPs
        tif_path = raster_dir / f'{layer_name}.tif'
        warp_with_gcps(png_path, tif_path, gcps)
        
        # Convert to shapefile
        shp_path = vector_dir / f'{layer_name}.shp'
        raster2shp(tif_path, shp_path, capacity)
        
        # Load shapefile for combination
        try:
            gdf = gpd.read_file(shp_path)
            gdf = gdf[gdf['val_kwhm3'] != 0]  # Remove background
            if not gdf.empty:
                gdf_list.append(gdf)
                print(f"✓ Added {len(gdf)} polygons from {layer_name}")
        except Exception as e:
            print(f"⚠️  Could not load {shp_path}: {e}")
    
    # Combine all layers into final shapefile
    if gdf_list:
        print("\n--- Creating Final Combined Shapefile ---")
        combined_gdf = pd.concat(gdf_list, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined_gdf, crs="EPSG:4326")
        
        # Fix invalid geometries
        combined_gdf['geometry'] = combined_gdf.geometry.buffer(0)
        
        # Save final shapefile
        final_path = results_dir / 'ukraine_salt_caverns.shp'
        combined_gdf.to_file(final_path)
        
        print(f"✅ SUCCESS: Created final shapefile with {len(combined_gdf)} polygons")
        print(f"📁 Saved to: {final_path}")
        
        # Print summary
        print("\n--- Summary ---")
        capacity_summary = combined_gdf.groupby('val_kwhm3').size()
        for capacity, count in capacity_summary.items():
            print(f"  {capacity} kWh/m³: {count} polygons")
        
        total_area_km2 = combined_gdf.to_crs('EPSG:3395').area.sum() / 1e6
        print(f"  Total area: {total_area_km2:.1f} km²")
        
        # Clean up temporary files
        print("\n--- Cleaning up temporary files ---")
        try:
            import shutil
            shutil.rmtree(raster_dir)
            shutil.rmtree(vector_dir)
            print("✓ Removed temporary raster and vector files")
        except Exception as e:
            print(f"⚠️  Could not remove temporary files: {e}")
            print(f"   You can manually delete: {raster_dir} and {vector_dir}")
        
        return final_path
    else:
        print("❌ No valid polygons found")
        return None


def main():
    """Main function with user input"""
    print("=== Ukraine Salt Cavern Blue Threshold Processor ===")
    
    # Default image path
    default_image_path = r"C:\Users\ls2823\OneDrive - Imperial College London\0_PHD\Repositories\salt_cave-gas_network\gas_network_exploring\salt_cave\original\salt_cave_ua.jpg"
    
    # Get input parameters
    image_input = input(f"Enter path to Ukraine salt cavern image (or press Enter for default): ").strip().strip('"')
    if not image_input:
        image_path = default_image_path
        print(f"Using default image: {image_path}")
    else:
        image_path = image_input
    
    # Try to find GCP file automatically
    image_dir = Path(image_path).parent
    gcp_file = image_dir / f"{Path(image_path).stem}_GCP_Points.json"
    
    if gcp_file.exists():
        print(f"Found GCP file: {gcp_file}")
        gcp_file_path = str(gcp_file)
    else:
        gcp_file_path = input("Enter path to GCP JSON file: ").strip().strip('"')
    
    output_dir = input("Enter output directory (or press Enter for 'ukraine_results'): ").strip()
    if not output_dir:
        output_dir = "ukraine_results"
    
    total_twh_input = input("Enter total TWh for Ukraine (89.8 TWh for validation, or press Enter to skip): ").strip()
    total_twh = float(total_twh_input) if total_twh_input else None
    
    try:
        result = process_ukraine_salt_caverns(image_path, gcp_file_path, output_dir, total_twh)
        if result:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📁 Final shapefile: {result}")
        else:
            print("\n❌ Processing failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
