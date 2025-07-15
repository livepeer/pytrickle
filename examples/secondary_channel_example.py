"""
Example demonstrating secondary publish channel usage.

This example shows how to use the secondary publish channel to send JSON metadata
alongside video processing. The secondary channel can be configured as either:
- "text" type: for sending JSON messages
- "video" type: for sending additional video/audio segments

This specific example uses the text type to send frame processing metadata.
"""

import asyncio
import logging
import torch
import time
from pytrickle import TrickleClient, VideoFrame, VideoOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_processor(frame: VideoFrame) -> VideoOutput:
    """
    Enhanced frame processor that adds effects and prepares metadata for secondary channel.
    
    This processor adds a blue tint to frames and returns metadata about the processing.
    """
    # Clone the tensor to avoid modifying the original
    tensor = frame.tensor.clone()
    
    # Add blue tint effect
    blue_intensity = 0.3
    tensor[:, :, :, 2] = torch.clamp(tensor[:, :, :, 2] + blue_intensity, 0, 1)
    
    # Create new frame with the processed tensor
    new_frame = frame.replace_tensor(tensor)
    
    # Create output with frame and request ID
    output = VideoOutput(new_frame, "enhanced_processor")
    
    return output

class SecondaryChannelClient:
    """Extended client that demonstrates secondary channel usage."""
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        secondary_publish_url: str,
        secondary_publish_type: str = "text"
    ):
        self.client = TrickleClient(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            secondary_publish_url=secondary_publish_url,
            secondary_publish_type=secondary_publish_type,
            width=704,
            height=384,
            frame_processor=self.process_frame_sync
        )
        self.frame_count = 0
        
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Synchronous frame processor that prepares metadata for async sending."""
        self.frame_count += 1
        
        # Process the frame
        output = enhanced_processor(frame)
        
        # Store metadata for async sending
        metadata = {
            "effect_applied": "blue_tint",
            "intensity": 0.3,
            "frame_shape": list(frame.tensor.shape),
            "processing_timestamp": int(time.time() * 1000),
            "frame_id": getattr(frame, 'frame_id', f'frame_{self.frame_count}')
        }
        
        # Send metadata asynchronously (fire and forget)
        asyncio.create_task(self._send_frame_metadata(metadata))
        
        return output
    
    async def _send_frame_metadata(self, metadata: dict):
        """Send frame metadata to secondary channel."""
        try:
            metadata_message = {
                "type": "frame_processed",
                "frame_number": self.frame_count,
                "timestamp": int(time.time() * 1000),
                "processing_info": metadata
            }
            
            # Send to secondary channel
            await self.client.send_secondary_text(metadata_message)
            logger.debug(f"Sent metadata for frame {self.frame_count} to secondary channel")
            
            # Also send periodic status updates
            if self.frame_count % 30 == 0:  # Every ~1 second at 30fps
                status_message = {
                    "type": "processing_status",
                    "frames_processed": self.frame_count,
                    "timestamp": int(time.time() * 1000),
                    "status": "active"
                }
                await self.client.send_secondary_text(status_message)
                logger.info(f"Sent status update: {self.frame_count} frames processed")
                
        except Exception as e:
            logger.error(f"Error sending metadata: {e}")
    
    async def start(self, request_id: str = "secondary_channel_demo"):
        """Start the client with secondary channel."""
        try:
            await self.client.start(request_id)
        except Exception as e:
            logger.error(f"Error in client: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the client."""
        if self.client:
            await self.client.stop()

async def main():
    """Main function demonstrating secondary channel usage."""
    # Configure trickle URLs
    subscribe_url = "http://localhost:3389/sample"
    publish_url = "http://localhost:3389/sample-output"
    secondary_publish_url = "http://localhost:3389/sample-metadata"  # Secondary channel for JSON metadata
    
    # Create client with secondary channel
    client = SecondaryChannelClient(
        subscribe_url=subscribe_url,
        publish_url=publish_url,
        secondary_publish_url=secondary_publish_url,
        secondary_publish_type="text"  # Use text type for JSON messages
    )
    
    logger.info("Starting stream processing with secondary channel...")
    logger.info(f"Main video: {subscribe_url} -> {publish_url}")
    logger.info(f"Metadata channel: {secondary_publish_url} (type: text)")
    
    try:
        await client.start("secondary_channel_demo")
    except KeyboardInterrupt:
        logger.info("Stream processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during stream processing: {e}")
    finally:
        logger.info("Stream processing stopped")

if __name__ == "__main__":
    asyncio.run(main()) 