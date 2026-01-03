"""
Enhanced class indices management for Plant Disease Detection system.

This script creates and manages class indices based on the model metadata,
ensuring consistency between training and inference.
"""

import os
import numpy as np
import json

def create_complete_class_indices():
    """Create a class indices dictionary based on the model metadata."""
    # Check if model metadata exists
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        classes = metadata.get('classes', [])
        if classes:
            # Create class_indices dictionary
            class_indices = {class_name: i for i, class_name in enumerate(classes)}
            
            # Save class indices
            np.save('class_indices.npy', class_indices)
            print(f" Created class_indices.npy with {len(class_indices)} classes from model metadata.")
            
            # Print some sample classes for verification
            print(f"\n📋 Sample classes (first 10):")
            for i, class_name in enumerate(list(class_indices.keys())[:10]):
                print(f"  {i:2d}. {class_name}")
            
            if len(class_indices) > 10:
                print(f"  ... and {len(class_indices) - 10} more classes")
                
            return class_indices
        else:
            print(" No classes found in model metadata.")
            return create_fallback_class_indices()
    else:
        print(" Could not find model_metadata.json")
        print("Creating fallback class indices based on expected dataset structure...")
        return create_fallback_class_indices()

def create_fallback_class_indices():
    """Create fallback class indices when metadata is not available."""
    # Complete class list based on PlantVillage dataset structure
    expected_classes = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___healthy',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___healthy',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___healthy',
        'Potato___Late_blight',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___healthy',
        'Strawberry___Leaf_scorch',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___healthy',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]
    
    class_indices = {class_name: i for i, class_name in enumerate(expected_classes)}
    np.save('class_indices.npy', class_indices)
    
    print(f" Created fallback class_indices.npy with {len(class_indices)} classes.")
    print(f"  Note: This is based on expected PlantVillage structure.")
    print(f"   If you have different classes, please retrain the model.")
    
    # Print sample classes
    print(f"\n📋 Sample classes (first 10):")
    for i, class_name in enumerate(expected_classes[:10]):
        plant_name = class_name.split('___')[0]
        disease_name = class_name.split('___')[1] if '___' in class_name else 'Unknown'
        print(f"  {i:2d}. {plant_name} - {disease_name}")
    
    print(f"  ... and {len(expected_classes) - 10} more classes")
    
    return class_indices

def verify_class_indices():
    """Verify that class indices file exists and is valid."""
    if not os.path.exists('class_indices.npy'):
        print(" class_indices.npy not found!")
        return False
    
    try:
        class_indices = np.load('class_indices.npy', allow_pickle=True).item()
        
        if not isinstance(class_indices, dict):
            print(" class_indices.npy is not a valid dictionary!")
            return False
        
        print(f" class_indices.npy is valid with {len(class_indices)} classes.")
        
        # Check for expected structure
        sample_keys = list(class_indices.keys())[:5]
        print(f"📋 Sample classes: {sample_keys}")
        
        # Verify indices are consecutive
        indices = sorted(class_indices.values())
        if indices != list(range(len(indices))):
            print("  Warning: Class indices are not consecutive!")
        
        return True
        
    except Exception as e:
        print(f" Error loading class_indices.npy: {e}")
        return False

def analyze_dataset_structure(dataset_path='dataset/PlantVillage'):
    """Analyze the actual dataset structure and create appropriate class indices."""
    color_dir = os.path.join(dataset_path, 'color')
    
    if not os.path.exists(color_dir):
        print(f" Dataset directory not found: {color_dir}")
        print("Please ensure the PlantVillage dataset is properly extracted.")
        return None
    
    # Get actual class directories
    actual_classes = []
    for class_dir in os.listdir(color_dir):
        class_path = os.path.join(color_dir, class_dir)
        if os.path.isdir(class_path):
            # Count images in this class
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if image_count > 0:  # Only include classes with images
                actual_classes.append(class_dir)
    
    actual_classes.sort()  # Sort for consistency
    
    if not actual_classes:
        print("No valid class directories found in dataset!")
        return None
    
    print(f" Found {len(actual_classes)} classes in dataset:")
    
    # Group by plant type for better display
    plant_groups = {}
    for class_name in actual_classes:
        if '___' in class_name:
            plant = class_name.split('___')[0]
            disease = class_name.split('___')[1]
            if plant not in plant_groups:
                plant_groups[plant] = []
            plant_groups[plant].append(disease)
    
    for plant, diseases in sorted(plant_groups.items()):
        print(f"  {plant}: {len(diseases)} conditions")
        for disease in sorted(diseases)[:3]:  # Show first 3 diseases
            print(f"    • {disease}")
        if len(diseases) > 3:
            print(f"    • ... and {len(diseases) - 3} more")
    
    # Create class indices from actual dataset
    class_indices = {class_name: i for i, class_name in enumerate(actual_classes)}
    
    return class_indices, actual_classes

def update_class_indices_from_dataset():
    """Update class indices based on actual dataset structure."""
    result = analyze_dataset_structure()
    
    if result is None:
        print(" Could not analyze dataset structure.")
        return False
    
    class_indices, actual_classes = result
    
    # Save the updated class indices
    np.save('class_indices.npy', class_indices)
    print(f"\n Updated class_indices.npy with {len(class_indices)} classes from dataset.")
    
    # Update model metadata if it exists
    if os.path.exists('model_metadata.json'):
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            metadata['classes'] = actual_classes
            metadata['num_classes'] = len(actual_classes)
            
            with open('model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(" Updated model_metadata.json with actual dataset classes.")
            
        except Exception as e:
            print(f"  Could not update model_metadata.json: {e}")
    
    return True

def main():
    """Main function to create or verify class indices."""
    print("PLANT DISEASE DETECTION - CLASS INDICES MANAGEMENT")
    print("=" * 60)
    
    # Check if we should analyze the dataset first
    if os.path.exists('dataset/PlantVillage/color'):
        print(" Dataset found. Analyzing structure...")
        success = update_class_indices_from_dataset()
        if success:
            return
        else:
            print("  Falling back to metadata-based approach...")
    
    # Try to create from metadata
    class_indices = create_complete_class_indices()
    
    # Verify the result
    if verify_class_indices():
        print("\n Class indices setup completed successfully!")
    else:
        print("\n Class indices setup failed!")
        return
    
    # Final verification
    print(f"\n📋 Final Summary:")
    print(f"  • Class indices file: class_indices.npy")
    print(f"  • Number of classes: {len(class_indices)}")
    print(f"  • Ready for model training/inference: ")

if __name__ == "__main__":
    main()