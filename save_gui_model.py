#!/usr/bin/env python3
"""
🔧 Model Converter for Siamese GUI
Converts training checkpoints to GUI-compatible model files

This script helps convert the full training checkpoints (with optimizer, scheduler, etc.)
to simplified model files that work seamlessly with the GUI application.
"""

import torch
import argparse
import os

def convert_checkpoint_to_gui_model(checkpoint_path, output_path=None):
    """
    Convert a training checkpoint to a GUI-compatible model file
    
    Args:
        checkpoint_path (str): Path to the training checkpoint (.pth file)
        output_path (str): Output path for GUI model (optional)
    """
    
    try:
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        
        # Load the full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract just the model state dict and essential info
        gui_model = {
            'model_state_dict': checkpoint['model_state_dict'],
            'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'model_info': {
                'architecture': 'ImprovedSiameseGCCapsuleNet',
                'embedding_dim': 128,
                'input_size': '50x50x3',
                'created_for': 'GUI Application'
            }
        }
        
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            output_path = f"{base_name}_gui_compatible.pth"
        
        # Save the GUI-compatible model
        print(f"💾 Saving GUI-compatible model to: {output_path}")
        torch.save(gui_model, output_path)
        
        print("✅ Conversion completed successfully!")
        print(f"📊 Model info:")
        print(f"   Best Validation Accuracy: {gui_model['best_val_acc']}")
        print(f"   Training Epoch: {gui_model['epoch']}")
        print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"\n🎯 You can now use '{output_path}' in the GUI application!")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error converting checkpoint: {e}")
        return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Convert Siamese training checkpoint to GUI-compatible model')
    parser.add_argument('checkpoint', help='Path to training checkpoint (.pth file)')
    parser.add_argument('-o', '--output', help='Output path for GUI model (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file '{args.checkpoint}' not found!")
        return
    
    convert_checkpoint_to_gui_model(args.checkpoint, args.output)

if __name__ == "__main__":
    # If run directly, try to convert common checkpoint names
    common_checkpoints = [
        'best_siamese_gc_capsule_model.pth',
        'checkpoint_epoch_25.pth',
        'checkpoint_epoch_20.pth',
        'checkpoint_epoch_15.pth'
    ]
    
    found_checkpoint = None
    for checkpoint in common_checkpoints:
        if os.path.exists(checkpoint):
            found_checkpoint = checkpoint
            break
    
    if found_checkpoint:
        print(f"🔍 Found checkpoint: {found_checkpoint}")
        convert_checkpoint_to_gui_model(found_checkpoint)
    else:
        print("❌ No common checkpoint files found in current directory.")
        print("📋 Looking for:")
        for checkpoint in common_checkpoints:
            print(f"   - {checkpoint}")
        print("\n💡 Usage: python save_gui_model.py <checkpoint_path>")
        print("   Example: python save_gui_model.py best_siamese_gc_capsule_model.pth") 