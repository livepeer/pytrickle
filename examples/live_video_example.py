#!/usr/bin/env python3
"""
Example demonstrating the live-video-to-video endpoint in pytrickle.

This shows how to use the new /live-video-to-video endpoint which is an alias
for the standard stream start endpoint but with a more descriptive name for
live video processing workflows.
"""

import asyncio
import aiohttp
import json

async def start_live_video_processing():
    """Example of starting live video processing using the new endpoint."""
    
    # Configuration for the live video stream
    config = {
        "subscribe_url": "http://localhost:3389/input",
        "publish_url": "http://localhost:3389/output", 
        "control_url": "http://localhost:3389/control",
        "events_url": "http://localhost:3389/events",
        "gateway_request_id": "live-video-example-001",
        "params": {
            "width": 1280,
            "height": 720,
            "effect": "live_enhancement",
            "quality": "high"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        # Start live video processing using the new endpoint
        async with session.post(
            "http://localhost:8080/live-video-to-video",
            json=config
        ) as response:
            result = await response.json()
            print(f"Live video processing started: {json.dumps(result, indent=2)}")
            
        # Example of updating parameters during live processing
        params_update = {
            "params": {
                "width": 1920,
                "height": 1080,
                "effect": "enhanced_clarity",
                "brightness": 1.2
            }
        }
        
        async with session.post(
            "http://localhost:8080/api/stream/params",
            json=params_update
        ) as response:
            result = await response.json()
            print(f"Parameters updated: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(start_live_video_processing())
