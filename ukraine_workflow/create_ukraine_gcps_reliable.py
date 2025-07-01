"""
Alternative GCP Creator with menu-based saving (more reliable)

This version uses a menu system instead of key presses for better reliability.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import threading
import time

class ReliableGCPCreator:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize image if too large for display
        height, width = self.image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.display_image = cv2.resize(self.image, (new_width, new_height))
            self.scale_factor = scale
        else:
            self.display_image = self.image.copy()
            self.scale_factor = 1.0
        
        self.gcps = []
        self.running = True
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.running:
            # Convert display coordinates back to original image coordinates
            orig_x = int(x / self.scale_factor)
            orig_y = int(y / self.scale_factor)
            
            # Draw point on display image
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"GCP {len(self.gcps)+1}", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Ukraine Salt Cavern Image - Click to add GCP', self.display_image)
            
            print(f"\n=== Adding GCP {len(self.gcps)+1} ===")
            print(f"Clicked at image coordinates: ({orig_x}, {orig_y})")
            print("Please enter the real-world coordinates for this point:")
            
            try:
                lat = float(input("Latitude (decimal degrees, e.g., 50.4501): "))
                lon = float(input("Longitude (decimal degrees, e.g., 30.5234): "))
                description = input("Description (e.g., 'Kiev center', 'Odessa port'): ")
                
                gcp = {
                    "image_x": orig_x,
                    "image_y": orig_y,
                    "latitude": lat,
                    "longitude": lon,
                    "description": description
                }
                
                self.gcps.append(gcp)
                print(f"✓ Added GCP {len(self.gcps)}: {gcp['description']}")
                print(f"  Image: ({orig_x}, {orig_y}) -> World: ({lat:.4f}, {lon:.4f})")
                
                # Show menu after each GCP
                self.show_menu()
                
            except ValueError:
                print("Invalid coordinates entered. Point not added.")
                self.redraw_image()
    
    def redraw_image(self):
        """Redraw image with all current GCPs"""
        self.display_image = self.image.copy()
        if self.scale_factor != 1.0:
            height, width = self.image.shape[:2]
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.display_image = cv2.resize(self.display_image, (new_width, new_height))
        
        for i, gcp in enumerate(self.gcps):
            x = int(gcp["image_x"] * self.scale_factor)
            y = int(gcp["image_y"] * self.scale_factor)
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"GCP {i+1}", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def show_menu(self):
        """Show menu options"""
        print(f"\n=== Menu (Current GCPs: {len(self.gcps)}) ===")
        print("1. Add another GCP (click on image)")
        print("2. Save GCPs and exit")
        print("3. Remove last GCP")
        print("4. List all GCPs")
        print("5. Quit without saving")
        
        if len(self.gcps) < 4:
            print(f"⚠️  Need at least 4 GCPs (currently have {len(self.gcps)})")
        else:
            print(f"✓ Ready to save ({len(self.gcps)} GCPs)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    print("Click on the image to add another GCP...")
                    break
                elif choice == '2':
                    if len(self.gcps) >= 4:
                        if self.save_gcps():
                            self.running = False
                            return
                        else:
                            print("Save failed. Try again.")
                    else:
                        print(f"Cannot save: Need at least 4 GCPs (have {len(self.gcps)})")
                elif choice == '3':
                    if self.gcps:
                        removed = self.gcps.pop()
                        print(f"Removed GCP: {removed['description']}")
                        self.redraw_image()
                        cv2.imshow('Ukraine Salt Cavern Image - Click to add GCP', self.display_image)
                        break
                    else:
                        print("No GCPs to remove")
                elif choice == '4':
                    self.list_gcps()
                elif choice == '5':
                    print("Quitting without saving...")
                    self.running = False
                    return
                else:
                    print("Invalid choice. Enter 1-5.")
            except KeyboardInterrupt:
                print("\nQuitting...")
                self.running = False
                return
    
    def list_gcps(self):
        """List all current GCPs"""
        if not self.gcps:
            print("No GCPs added yet.")
        else:
            print(f"\n=== Current GCPs ({len(self.gcps)}) ===")
            for i, gcp in enumerate(self.gcps):
                print(f"{i+1}. {gcp['description']}")
                print(f"   Image: ({gcp['image_x']}, {gcp['image_y']})")
                print(f"   World: ({gcp['latitude']:.4f}, {gcp['longitude']:.4f})")
    
    def save_gcps(self):
        """Save GCPs to JSON file"""
        try:
            output_file = self.image_path.parent / f"{self.image_path.stem}_GCP_Points.json"
            
            print(f"Saving to: {output_file}")
            
            # Check if directory is writable
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True, exist_ok=True)
            
            gcp_data = {
                "image_file": str(self.image_path.name),
                "coordinate_system": "EPSG:4326",
                "gcps": self.gcps,
                "creation_info": {
                    "total_gcps": len(self.gcps),
                    "image_size": {
                        "width": self.image.shape[1],
                        "height": self.image.shape[0]
                    }
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(gcp_data, f, indent=2)
            
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"\n✅ SUCCESS: Saved {len(self.gcps)} GCPs to: {output_file}")
                print(f"   File size: {file_size} bytes")
                self.list_gcps()
                return True
            else:
                print(f"❌ File was not created: {output_file}")
                return False
                
        except PermissionError:
            print(f"❌ Permission denied: Cannot write to {output_file}")
            
            # Try fallback location
            fallback_file = Path.cwd() / f"{self.image_path.stem}_GCP_Points.json"
            print(f"Trying fallback: {fallback_file}")
            try:
                with open(fallback_file, 'w') as f:
                    json.dump(gcp_data, f, indent=2)
                print(f"✅ Saved to fallback location: {fallback_file}")
                return True
            except Exception as e:
                print(f"❌ Fallback failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return False
    
    def create_gcps(self):
        """Main GCP creation function"""
        print("=== Reliable Ukraine Salt Cavern GCP Creator ===")
        print("\nThis version uses a menu system for better reliability.")
        print("\nGood reference points for Ukraine:")
        print("- Major cities: Kiev (50.4501, 30.5234), Kharkiv (49.9935, 36.2304)")
        print("- Odessa (46.4825, 30.7233), Lviv (49.8397, 24.0297)")
        print("- Border corners, coastline points, etc.")
        print("\nInstructions:")
        print("1. Click on recognizable points in the image")
        print("2. Enter coordinates and description for each point")
        print("3. Use the menu to save when you have at least 4 GCPs")
        
        cv2.namedWindow('Ukraine Salt Cavern Image - Click to add GCP', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Ukraine Salt Cavern Image - Click to add GCP', self.mouse_callback)
        cv2.imshow('Ukraine Salt Cavern Image - Click to add GCP', self.display_image)
        
        print("\nClick on the image to add your first GCP...")
        
        while self.running:
            # Check if window was closed
            if cv2.getWindowProperty('Ukraine Salt Cavern Image - Click to add GCP', cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow was closed.")
                if len(self.gcps) >= 4:
                    save_choice = input(f"You have {len(self.gcps)} GCPs. Save them? (y/n): ")
                    if save_choice.lower().startswith('y'):
                        self.save_gcps()
                break
            
            cv2.waitKey(30)  # Small delay
        
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("=== Reliable GCP Creator ===")
    image_path = input("Enter path to Ukraine salt cavern image: ").strip().strip('"')
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return
    
    try:
        creator = ReliableGCPCreator(image_path)
        creator.create_gcps()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
