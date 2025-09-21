#!/usr/bin/env python3
"""
YouTube Video Downloader and Frame Extractor
============================================

This script performs two main stages:
1. Downloads YouTube videos from URLs in youtube-urls.txt
2. Extracts first 2 minutes of frames at 2 FPS as PNG files

Uses Streamlit for progress tracking and user interface.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
import yt_dlp
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional
import time
from pathlib import Path

class VideoDownloaderApp:
    """Main application class for video downloading and frame extraction"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.urls_file = self.base_dir / "youtube-urls.txt"
        self.downloaded_videos_dir = self.base_dir / "downloaded_videos"
        self.extracted_frames_dir = self.base_dir / "extracted_frames"
        self.labelling_images_dir = self.base_dir / "labelling_images"
        
        # Create directories if they don't exist
        self.downloaded_videos_dir.mkdir(exist_ok=True)
        self.extracted_frames_dir.mkdir(exist_ok=True)
        self.labelling_images_dir.mkdir(exist_ok=True)
        
        # Track errors and progress
        self.errors = []
        self.completed_downloads = []
        self.completed_extractions = []
    
    def get_video_id_from_url(self, youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            parsed_url = urlparse(youtube_url)
            if 'v' in parse_qs(parsed_url.query):
                return parse_qs(parsed_url.query)['v'][0]
        except Exception:
            pass
        return None
    
    def parse_time_string(self, time_str: str) -> float:
        """Parse time string (e.g., '2m', '30s', '1m30s') to seconds"""
        if not time_str or time_str == "0":
            return 0.0
        
        time_str = time_str.lower().strip()
        total_seconds = 0.0
        
        # Handle formats like "2m30s", "1m", "30s"
        import re
        
        # Extract minutes
        minutes_match = re.search(r'(\d+)m', time_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60
        
        # Extract seconds
        seconds_match = re.search(r'(\d+)s', time_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))
        
        # If no units specified, assume seconds
        if not minutes_match and not seconds_match:
            try:
                total_seconds = float(time_str)
            except ValueError:
                raise ValueError(f"Invalid time format: {time_str}")
        
        return total_seconds
    
    def parse_video_time_string(self, video_time_str: str) -> tuple[Optional[str], float]:
        """Parse video:time string (e.g., 'ae94835:2m') to video_id and start_time_seconds"""
        if not video_time_str or video_time_str.strip() == "0s":
            return None, 0.0
        
        video_time_str = video_time_str.strip()
        
        # Check if it contains a colon (videoid:time format)
        if ':' not in video_time_str:
            raise ValueError(f"Invalid format. Expected 'videoid:time' or '0s', got: {video_time_str}")
        
        try:
            video_id, time_part = video_time_str.split(':', 1)
            video_id = video_id.strip()
            time_part = time_part.strip()
            
            if not video_id:
                raise ValueError("Video ID cannot be empty")
            
            start_time_seconds = self.parse_time_string(time_part)
            return video_id, start_time_seconds
            
        except ValueError as e:
            if "Invalid format" in str(e):
                raise e
            raise ValueError(f"Invalid videoid:time format: {video_time_str}. Error: {str(e)}")
    
    def load_youtube_urls(self) -> List[str]:
        """Load YouTube URLs from the text file"""
        if not self.urls_file.exists():
            st.error(f"File not found: {self.urls_file}")
            return []
        
        try:
            with open(self.urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return urls
        except Exception as e:
            st.error(f"Error reading URLs file: {e}")
            return []
    
    def download_video(self, youtube_url: str, progress_bar) -> Optional[str]:
        """Download a single YouTube video using yt-dlp"""
        video_id = self.get_video_id_from_url(youtube_url)
        if not video_id:
            raise Exception(f"Could not extract video ID from URL: {youtube_url}")
        
        # Check if video already exists
        video_path = self.downloaded_videos_dir / f"{video_id}.mp4"
        if video_path.exists():
            st.info(f"Video {video_id} already exists, skipping download")
            return str(video_path)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[height<=720]/best[height<=480]/best',
            'outtmpl': str(self.downloaded_videos_dir / f"{video_id}.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
            'socket_timeout': 60,
            'retries': 3,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video
                ydl.download([youtube_url])
                
                # Find the downloaded file (could be .mp4, .webm, etc.)
                for ext in ['mp4', 'webm', 'mkv', 'avi']:
                    potential_path = self.downloaded_videos_dir / f"{video_id}.{ext}"
                    if potential_path.exists():
                        # Rename to .mp4 if it's not already
                        final_path = self.downloaded_videos_dir / f"{video_id}.mp4"
                        if potential_path != final_path:
                            potential_path.rename(final_path)
                        return str(final_path)
                
                raise Exception("Downloaded file not found")
                
        except Exception as e:
            raise Exception(f"Failed to download video {video_id}: {str(e)}")
    
    def extract_frames_from_video(self, video_path: str, progress_bar, start_time_seconds: float = 0.0) -> int:
        """Extract first 2 minutes of frames at 2 FPS as PNG files starting from specified time"""
        video_id = Path(video_path).stem
        frames_dir = self.extracted_frames_dir / video_id
        frames_dir.mkdir(exist_ok=True)
        
        # Check if frames already exist
        existing_frames = list(frames_dir.glob("*.png"))
        if existing_frames:
            st.info(f"Frames for {video_id} already exist ({len(existing_frames)} files), skipping extraction")
            return len(existing_frames)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback
            
            # Seek to start time
            start_frame = int(start_time_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calculate frame interval for 2 FPS
            frame_interval = int(fps / 2)  # Every 0.5 seconds
            max_frames = 2 * 60 * 2  # 2 minutes at 2 FPS = 240 frames
            
            frame_count = start_frame
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at the specified interval
                if (frame_count - start_frame) % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Save frame as PNG
                    frame_filename = f"frame_{timestamp:06.2f}s.png"
                    frame_path = frames_dir / frame_filename
                    
                    # Convert BGR to RGB for proper color
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    
                    extracted_count += 1
                    
                    # Update progress
                    progress = extracted_count / max_frames
                    progress_bar.progress(progress)
                
                frame_count += 1
                
                # Stop after 2 minutes from start time
                if (frame_count - start_frame) / fps >= 120:  # 2 minutes
                    break
            
            return extracted_count
            
        finally:
            cap.release()
    
    def extract_frames_single_video(self, video_id: str, start_time_seconds: float, progress_bar) -> int:
        """Extract frames from a single video starting at specified time"""
        # Check if video file exists
        video_path = self.downloaded_videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise Exception(f"Video file not found: {video_path}")
        
        frames_dir = self.extracted_frames_dir / video_id
        frames_dir.mkdir(exist_ok=True)
        
        # Check if frames already exist
        existing_frames = list(frames_dir.glob("*.png"))
        if existing_frames:
            st.info(f"Frames for {video_id} already exist ({len(existing_frames)} files), skipping extraction")
            return len(existing_frames)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback
            
            # Seek to start time
            start_frame = int(start_time_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calculate frame interval for 2 FPS
            frame_interval = int(fps / 2)  # Every 0.5 seconds
            max_frames = 2 * 60 * 2  # 2 minutes at 2 FPS = 240 frames
            
            frame_count = start_frame
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at the specified interval
                if (frame_count - start_frame) % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Save frame as PNG
                    frame_filename = f"frame_{timestamp:06.2f}s.png"
                    frame_path = frames_dir / frame_filename
                    
                    # Convert BGR to RGB for proper color
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    
                    extracted_count += 1
                    
                    # Update progress
                    progress = extracted_count / max_frames
                    progress_bar.progress(progress)
                
                frame_count += 1
                
                # Stop after 2 minutes from start time
                if (frame_count - start_frame) / fps >= 120:  # 2 minutes
                    break
            
            return extracted_count
            
        finally:
            cap.release()
    
    def extract_labelling_frame(self, video_path: str, frame_number: int) -> str:
        """Extract a single frame from video at appropriate time for labelling"""
        video_id = Path(video_path).stem
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps
            
            # Determine extraction time based on video duration
            if duration_seconds < 60:  # Under 1 minute
                extract_time = 30  # Extract at 30 seconds
            elif duration_seconds < 120:  # Under 2 minutes
                extract_time = 60  # Extract at 1 minute
            else:
                extract_time = 120  # Extract at 2 minutes
            
            # Ensure we don't seek beyond video duration
            extract_time = min(extract_time, duration_seconds - 1)
            
            # Seek to extraction time
            target_frame = int(extract_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            if not ret:
                raise Exception(f"Could not read frame at {extract_time}s from video {video_id}")
            
            # Save frame as PNG with zero-padded incremental numbering
            frame_filename = f"{frame_number:02d}_{video_id}.png"
            frame_path = self.labelling_images_dir / frame_filename
            
            # Convert BGR to RGB for proper color and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            return str(frame_path)
            
        finally:
            cap.release()
    
    def run_stage_1(self) -> List[str]:
        """Run Stage 1: Download all videos"""
        st.header("🎬 Stage 1: Downloading Videos")
        
        urls = self.load_youtube_urls()
        if not urls:
            return []
        
        downloaded_paths = []
        
        for i, url in enumerate(urls):
            video_id = self.get_video_id_from_url(url)
            st.subheader(f"Downloading video {i+1}/{len(urls)}: {video_id or 'Unknown ID'}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Starting download...")
                progress_bar.progress(0.1)
                
                video_path = self.download_video(url, progress_bar)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Download completed!")
                
                downloaded_paths.append(video_path)
                self.completed_downloads.append(video_id or url)
                
            except Exception as e:
                error_msg = f"❌ Error downloading {video_id or url}: {str(e)}"
                status_text.text(error_msg)
                self.errors.append(error_msg)
                st.error(error_msg)
            
            time.sleep(0.5)  # Brief pause between downloads
        
        return downloaded_paths
    
    def run_stage_2(self, video_paths: List[str], start_time_seconds: float = 0.0):
        """Run Stage 2: Extract frames from downloaded videos"""
        st.header("🖼️ Stage 2: Extracting Frames")
        
        if not video_paths:
            st.warning("No videos to process for frame extraction")
            return
        
        for i, video_path in enumerate(video_paths):
            video_id = Path(video_path).stem
            st.subheader(f"Extracting frames {i+1}/{len(video_paths)}: {video_id}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Extracting frames...")
                
                frame_count = self.extract_frames_from_video(video_path, progress_bar, start_time_seconds)
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Extracted {frame_count} frames!")
                
                self.completed_extractions.append(video_id)
                
            except Exception as e:
                error_msg = f"❌ Error extracting frames from {video_id}: {str(e)}"
                status_text.text(error_msg)
                self.errors.append(error_msg)
                st.error(error_msg)
            
            time.sleep(0.5)  # Brief pause between extractions
    
    def run_stage_3(self):
        """Run Stage 3: Extract labelling frames from all downloaded videos"""
        st.header("🖼️ Stage 3: Extracting Labelling Frames")
        
        # Get all downloaded video files
        video_files = list(self.downloaded_videos_dir.glob("*.mp4"))
        
        if not video_files:
            st.warning("No downloaded videos found. Please run Stage 1 first to download videos.")
            return
        
        # Clear existing labelling frames (overwrite as requested)
        existing_frames = list(self.labelling_images_dir.glob("*.png"))
        if existing_frames:
            st.info(f"Removing {len(existing_frames)} existing labelling frames...")
            for frame_file in existing_frames:
                frame_file.unlink()
        
        st.info(f"Processing {len(video_files)} videos for labelling frame extraction...")
        
        frame_number = 1  # Reset numbering each time
        successful_extractions = []
        
        for i, video_path in enumerate(video_files):
            video_id = video_path.stem
            st.subheader(f"Extracting labelling frame {i+1}/{len(video_files)}: {video_id}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Extracting labelling frame...")
                progress_bar.progress(0.5)
                
                frame_path = self.extract_labelling_frame(str(video_path), frame_number)
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Extracted labelling frame: {Path(frame_path).name}")
                
                successful_extractions.append(video_id)
                frame_number += 1
                
            except Exception as e:
                error_msg = f"❌ Error extracting labelling frame from {video_id}: {str(e)}"
                status_text.text(error_msg)
                self.errors.append(error_msg)
                st.error(error_msg)
            
            time.sleep(0.2)  # Brief pause between extractions
        
        # Show final summary
        if successful_extractions:
            st.success(f"Successfully extracted {len(successful_extractions)} labelling frames!")
            st.info(f"Frames saved to: {self.labelling_images_dir}")
        else:
            st.error("No labelling frames were extracted successfully.")
    
    def show_summary(self):
        """Display summary of completed work and errors"""
        st.header("📊 Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Completed Downloads")
            if self.completed_downloads:
                for video_id in self.completed_downloads:
                    st.text(f"• {video_id}")
            else:
                st.text("No downloads completed")
        
        with col2:
            st.subheader("✅ Completed Frame Extractions")
            if self.completed_extractions:
                for video_id in self.completed_extractions:
                    st.text(f"• {video_id}")
            else:
                st.text("No frame extractions completed")
        
        if self.errors:
            st.subheader("❌ Errors Encountered")
            for error in self.errors:
                st.error(error)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="YouTube Video Downloader & Frame Extractor",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 YouTube Video Downloader & Frame Extractor")
    st.markdown("Downloads YouTube videos and extracts frames at 2 FPS")
    
    app = VideoDownloaderApp()
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["📦 Bulk Processing", "🎯 Single Video Extraction", "🏷️ Labelling Frames"])
    
    with tab1:
        st.header("Bulk Processing - All Videos")
        st.markdown("Downloads all videos from youtube-urls.txt and extracts first 2 minutes of frames at 2 FPS from the beginning")
        
        # Add a button to start the bulk process
        if st.button("🚀 Start Bulk Processing", type="primary", key="bulk_process"):
            # Run both stages (always start from beginning for bulk processing)
            downloaded_videos = app.run_stage_1()
            app.run_stage_2(downloaded_videos, 0.0)
            
            # Show summary
            app.show_summary()
    
    with tab2:
        st.header("Single Video Extraction")
        st.markdown("Extract frames from a specific video starting at a specific time point")
        
        # Add video:time input
        st.subheader("⚙️ Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            video_time_input = st.text_input(
                "Video ID and Start Time",
                value="",
                placeholder="ae94835:2m",
                help="Format: videoid:time (e.g., 'ae94835:2m' to extract video ae94835 starting at 2 minutes)",
                key="single_video_time"
            )
        
        with col2:
            st.info("**Format Examples:**\n- `ae94835:0s` - Video ae94835 from beginning\n- `ae94835:30s` - Video ae94835 from 30 seconds\n- `abc123:2m` - Video abc123 from 2 minutes\n- `xyz789:1m30s` - Video xyz789 from 1 minute 30 seconds")
        
        # Validate video:time input
        if video_time_input.strip():
            try:
                video_id, start_time_seconds = app.parse_video_time_string(video_time_input)
                if video_id:
                    st.success(f"Video ID: {video_id}, Start time: {start_time_seconds} seconds")
                    
                    # Check if video exists
                    video_path = app.downloaded_videos_dir / f"{video_id}.mp4"
                    if video_path.exists():
                        st.info(f"✅ Video {video_id}.mp4 found")
                    else:
                        st.warning(f"⚠️ Video {video_id}.mp4 not found. Please download it first using bulk processing.")
                else:
                    st.info("Enter a video ID and time to extract frames")
            except ValueError as e:
                st.error(f"Invalid format: {e}")
                video_id, start_time_seconds = None, 0.0
        else:
            video_id, start_time_seconds = None, 0.0
        
        # Add button to start single video extraction
        if st.button("🎯 Extract Frames from Single Video", type="primary", key="single_extract", disabled=not video_id):
            if video_id:
                st.subheader(f"Extracting frames from {video_id}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Extracting frames...")
                    
                    frame_count = app.extract_frames_single_video(video_id, start_time_seconds, progress_bar)
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Extracted {frame_count} frames!")
                    
                    st.success(f"Successfully extracted {frame_count} frames from video {video_id} starting at {start_time_seconds} seconds")
                    
                except Exception as e:
                    error_msg = f"❌ Error extracting frames from {video_id}: {str(e)}"
                    status_text.text(error_msg)
                    st.error(error_msg)
    
    with tab3:
        st.header("Labelling Frame Extraction")
        st.markdown("Extract single frames from all downloaded videos for labelling purposes")
        
        # Show information about what this does
        st.info("**What this does:**\n"
               "- Processes all videos in the downloaded_videos directory\n"
               "- Extracts one frame per video at an appropriate time:\n"
               "  - 2 minutes for videos ≥ 2 minutes\n"
               "  - 1 minute for videos ≥ 1 minute but < 2 minutes\n"
               "  - 30 seconds for videos < 1 minute\n"
               "- Saves frames as zero-padded numbered PNG files: `01_videoid.png`, `02_videoid.png`, etc.\n"
               "- Overwrites existing labelling frames")
        
        # Show current video count
        video_files = list(app.downloaded_videos_dir.glob("*.mp4"))
        if video_files:
            st.success(f"Found {len(video_files)} videos ready for processing")
            
            # Show some example video IDs
            example_videos = [v.stem for v in video_files[:5]]
            st.text(f"Example videos: {', '.join(example_videos)}")
            if len(video_files) > 5:
                st.text(f"... and {len(video_files) - 5} more")
        else:
            st.warning("No videos found. Please run 'Bulk Processing' first to download videos.")
        
        # Add button to start Stage 3
        if st.button("🏷️ Extract Labelling Frames", type="primary", key="stage3_extract", disabled=len(video_files) == 0):
            app.run_stage_3()
    
    # Show current status in sidebar
    st.sidebar.header("📁 Directory Info")
    st.sidebar.text(f"URLs file: {app.urls_file}")
    st.sidebar.text(f"Downloads: {app.downloaded_videos_dir}")
    st.sidebar.text(f"Frames: {app.extracted_frames_dir}")
    st.sidebar.text(f"Labelling: {app.labelling_images_dir}")
    
    # Show existing files
    if app.downloaded_videos_dir.exists():
        existing_videos = list(app.downloaded_videos_dir.glob("*.mp4"))
        st.sidebar.text(f"Existing videos: {len(existing_videos)}")
        
        if existing_videos:
            st.sidebar.subheader("Available Videos")
            for video_path in existing_videos[:10]:  # Show first 10
                video_id = video_path.stem
                st.sidebar.text(f"• {video_id}")
            if len(existing_videos) > 10:
                st.sidebar.text(f"... and {len(existing_videos) - 10} more")
    
    if app.extracted_frames_dir.exists():
        existing_frame_dirs = [d for d in app.extracted_frames_dir.iterdir() if d.is_dir()]
        st.sidebar.text(f"Existing frame dirs: {len(existing_frame_dirs)}")
    
    if app.labelling_images_dir.exists():
        existing_labelling_frames = list(app.labelling_images_dir.glob("*.png"))
        st.sidebar.text(f"Existing labelling frames: {len(existing_labelling_frames)}")

if __name__ == "__main__":
    main()