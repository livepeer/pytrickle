def trickle_handler(handler_type: str):
    """
    Decorator to mark methods as trickle handlers.
    
    Args:
        handler_type: Type of handler ('video', 'audio', 'model_loader', 'param_updater', 'stream_stop')
    
    Usage:
        @trickle_handler("video")
        async def process_video(self, frame: VideoFrame) -> Optional[VideoFrame]:
            # Process video frame using existing VideoFrame interface
            return processed_frame
            
        @trickle_handler("audio") 
        async def process_audio(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
            # Process audio frame using existing AudioFrame interface
            return [processed_frame]
            
        @trickle_handler("model_loader")
        async def load_model(self, **kwargs):
            # Load model with keyword arguments
            pass
            
        @trickle_handler("param_updater")
        async def update_params(self, params: Dict[str, Any]):
            # Update parameters with dict
            pass
            
        @trickle_handler("stream_stop")
        async def on_stream_stop(self):
            # Handle stream stop
            pass
    """
    def decorator(func):
        func._trickle_handler_type = handler_type
        func._trickle_handler = True
        return func
    return decorator
