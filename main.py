import os
import argparse
from kitchen_safety.model_manager import ModelManager
from kitchen_safety.image_processing import process_test_images

def main():
    parser = argparse.ArgumentParser(description="Kitchen Safety Testing System")
    parser.add_argument("--mode", choices=["test", "webcam", "video"], default="test",
                        help="Processing mode (default: test)")
    parser.add_argument("--model", default="kitchen",
                        choices=["meat", "kitchen", "sausage"],
                        help="Detection model to use")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (0-1)")
    parser.add_argument("--input", default="test_images",
                        help="Path to input folder (for test mode)")
    parser.add_argument("--output", default="test_results",
                        help="Output directory for test mode or output file for video mode")
    parser.add_argument("--video", default="",
                        help="Path to video file (required for video mode)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable Gemini AI analysis")
    parser.add_argument("--no-hazard", action="store_true",
                        help="Disable hazard report generation")
    parser.add_argument("--no-gemini-fallback", action="store_true",
                        help="Disable Gemini fallback object detection")
    args = parser.parse_args()

    # Initialize model manager
    model_manager = ModelManager()

    if args.mode == "test":
        # In test mode, output is a directory
        os.makedirs(args.output, exist_ok=True)
        print(f"\nStarting test mode with images from: {args.input}")
        process_test_images(
            image_folder=args.input,
            model=args.model,
            confidence=args.confidence,
            output=args.output,
            enable_ai_analysis=not args.no_ai,
            hazard_report=not args.no_hazard,
            gemini_fallback=not args.no_gemini_fallback
        )

    elif args.mode == "webcam":
        print("\nStarting real-time kitchen monitoring")
        from kitchen_safety.webcam_processing import process_webcam
        process_webcam(
            model=args.model,
            confidence=args.confidence,
            enable_ai_analysis=not args.no_ai
        )

    elif args.mode == "video":
        if not args.video:
            print("Error: Video path required for video mode")
            return

        # For video mode, output is a file path.
        # Create parent directory for output file (not the file itself)
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessing video: {args.video}")
        from kitchen_safety.video_processing import process_prerecorded_video
        process_prerecorded_video(
            video_path=args.video,
            model=args.model,
            confidence=args.confidence,
            output=args.output,
            enable_ai_analysis=not args.no_ai
        )

if __name__ == "__main__":
    main()
