#!/usr/bin/env python3
"""
Example usage script for the AI Lip Reading project.

This script demonstrates how to use the lip reading functionality
with different configurations and provides sample code for integration.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import lipreading module
sys.path.append(str(Path(__file__).parent.parent))

from lipreading import lip_reading_basic, lip_reading_premium, detect_faces

def example_basic_usage():
    """Example of using the basic (offline) lip reading model."""
    print("=== Basic Lip Reading Example ===")
    
    # Example video file path (replace with your actual video file)
    video_file = "sample_video.mp4"
    
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found. Please provide a valid video file.")
        return
    
    try:
        # Run basic lip reading
        transcription = lip_reading_basic(
            video_file=video_file,
            language="en",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Error during lip reading: {e}")

def example_premium_usage():
    """Example of using the premium (cloud) lip reading model."""
    print("=== Premium Lip Reading Example ===")
    
    # Example video file path (replace with your actual video file)
    video_file = "sample_video.mp4"
    
    # API key for premium service (replace with your actual API key)
    api_key = "your_api_key_here"
    
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found. Please provide a valid video file.")
        return
    
    if api_key == "your_api_key_here":
        print("Please provide a valid API key for premium service.")
        return
    
    try:
        # Run premium lip reading
        transcription = lip_reading_premium(
            video_file=video_file,
            api_key=api_key,
            language="en"
        )
        
        print(f"Transcription: {transcription}")
        
    except Exception as e:
        print(f"Error during premium lip reading: {e}")

def example_face_detection():
    """Example of face detection functionality."""
    print("=== Face Detection Example ===")
    
    # Example video file path (replace with your actual video file)
    video_file = "sample_video.mp4"
    
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found. Please provide a valid video file.")
        return
    
    try:
        # Run face detection
        detect_faces(video_file)
        
    except Exception as e:
        print(f"Error during face detection: {e}")

def example_integration():
    """Example of integrating lip reading into a larger application."""
    print("=== Integration Example ===")
    
    class LipReadingApp:
        def __init__(self, model_type="basic", api_key=None):
            self.model_type = model_type
            self.api_key = api_key
            self.supported_languages = ["en", "es", "fr", "de"]
        
        def process_video(self, video_path, language="en"):
            """Process a video file and return transcription."""
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if language not in self.supported_languages:
                raise ValueError(f"Unsupported language: {language}")
            
            if self.model_type == "basic":
                return lip_reading_basic(video_path, language=language)
            elif self.model_type == "premium":
                if not self.api_key:
                    raise ValueError("API key required for premium model")
                return lip_reading_premium(video_path, self.api_key, language=language)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        def batch_process(self, video_directory):
            """Process multiple videos in a directory."""
            results = {}
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            
            for file in os.listdir(video_directory):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(video_directory, file)
                    try:
                        transcription = self.process_video(video_path)
                        results[file] = transcription
                    except Exception as e:
                        results[file] = f"Error: {e}"
            
            return results
    
    # Example usage of the integration class
    app = LipReadingApp(model_type="basic")
    
    # Process a single video
    try:
        transcription = app.process_video("sample_video.mp4")
        print(f"Single video transcription: {transcription}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Process multiple videos (if directory exists)
    video_dir = "videos"
    if os.path.exists(video_dir):
        results = app.batch_process(video_dir)
        print("Batch processing results:")
        for video, result in results.items():
            print(f"  {video}: {result}")

def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="AI Lip Reading Examples")
    parser.add_argument('--example', choices=['basic', 'premium', 'face_detection', 'integration', 'all'],
                       default='all', help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 'basic' or args.example == 'all':
        example_basic_usage()
        print()
    
    if args.example == 'premium' or args.example == 'all':
        example_premium_usage()
        print()
    
    if args.example == 'face_detection' or args.example == 'all':
        example_face_detection()
        print()
    
    if args.example == 'integration' or args.example == 'all':
        example_integration()
        print()

if __name__ == "__main__":
    main()
