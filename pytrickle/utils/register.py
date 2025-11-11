"""
Capability registration for PyTrickle.

Provides functionality to register worker capabilities with orchestrators,
enabling automatic discovery and scaling of AI processing workers.
"""

import os
import logging
from typing import Optional, Union
from urllib.parse import urlparse, ParseResult
import aiohttp
import asyncio

class RegisterCapability:
    """
    Register worker capabilities with orchestrators using environment variables.
    
    Environment variables:
        ORCH_URL: Orchestrator URL
        ORCH_SECRET: Orchestrator authentication secret
        CAPABILITY_NAME: Name of the capability
        CAPABILITY_DESCRIPTION: Description of the capability
        CAPABILITY_URL: URL where this capability can be reached
        CAPABILITY_CAPACITY: Maximum concurrent streams (default: 1)
        CAPABILITY_PRICE_PER_UNIT: Price per processing unit (default: 0)
        CAPABILITY_PRICE_SCALING: Price scaling factor (default: 1)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional logger, creates default if none provided."""
        self.logger = logger or logging.getLogger(__name__)

    def _build_register_request(self, **values) -> dict:
        """Build the registration request payload."""
        return {
            "url": values["capability_url"],
            "name": values["capability_name"],
            "description": values["capability_desc"],
            "capacity": values["capability_capacity"],
            "price_per_unit": values["capability_price_per_unit"],
            "price_scaling": values["capability_price_scaling"],
        }
    async def _make_registration_request(self, orch_url: str, orch_secret: str, register_req: dict,
                                 max_retries: int = 10, delay: float = 2.0, timeout: float = 5.0) -> Union[ParseResult, bool]:
        """Make the actual HTTP registration request with retry logic."""
        headers = {"Authorization": orch_secret, "Content-Type": "application/json"}
        
        self.logger.info(f"Registering capability: {register_req}")
        
        for attempt in range(1, max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{orch_url}/capability/register",
                        json=register_req,
                        headers=headers,
                        timeout=timeout,
                        ssl=False,
                    ) as resp:
                        if resp.status == 200:
                            name = register_req.get("name", "unknown")
                            self.logger.info(f"Capability {name!r} registered successfully")
                            return urlparse(register_req["url"])
                        elif resp.status == 400:
                            self.logger.error("Orchestrator secret incorrect")
                            return False
                        else:
                            self.logger.warning(f"Register attempt {attempt} failed: {resp.status} {await resp.text()}")

            except aiohttp.ClientConnectorError as e:
                # Handle connect call failed without raising
                self.logger.warning(f"Register attempt {attempt} failed to connect: {e}")
            except asyncio.TimeoutError as e:
                self.logger.warning(f"Register attempt {attempt} timed out: {e}")
            except aiohttp.ClientError as e:
                self.logger.warning(f"Register attempt {attempt} failed with client error: {e}")
            except Exception as e:
                # Ensure unexpected errors don't bubble up and fail startup
                self.logger.error(f"Register attempt {attempt} failed with unexpected exception: {e}")
                
            if attempt < max_retries:
                await asyncio.sleep(delay)
            else:
                self.logger.warning("All registration retries failed")
                return False
                
        return False
        
    async def register_capability(
        self,
        orch_url: Optional[str] = None,
        orch_secret: Optional[str] = None,
        capability_name: Optional[str] = None,
        capability_desc: Optional[str] = None,
        capability_capacity: Optional[int] = None,
        capability_price_per_unit: Optional[int] = None,
        capability_price_scaling: Optional[int] = None,
        max_retries: int = 3,
        delay: float = 2.0,
        timeout: float = 1.5
    ) -> Union[ParseResult, bool]:
        """
        Register this worker capability with the orchestrator.
        
        This method automatically uses environment variables as defaults when parameters are not provided.
        Environment variables are used in the following order: parameter value > environment variable > hardcoded default.
        
        Environment Variables Used as Defaults:
        - ORCH_URL: Orchestrator URL
        - ORCH_SECRET: Orchestrator secret
        - CAPABILITY_NAME: Capability name
        - CAPABILITY_DESCRIPTION: Capability description 
        - CAPABILITY_URL: Capability URL (always from environment, cannot be overridden)
        - CAPABILITY_CAPACITY: Max concurrent streams
        - CAPABILITY_PRICE_PER_UNIT: Price per unit 
        - CAPABILITY_PRICE_SCALING: Price scaling
        
        Additional Args:
        - max_retries: Maximum number of registration attempts (default: 10)
        - delay: Delay between retry attempts in seconds (default: 2.0)
        - timeout: Request timeout in seconds (default: 5.0)
            
        Returns:
            The registered capability URL (as ParseResult object) if registration succeeded, False otherwise
        """
        # Get values from env vars if not provided
        orch_url = orch_url or os.environ.get("ORCH_URL", "")
        # Pop secret from environment to reduce exposure
        orch_secret = orch_secret or os.environ.pop("ORCH_SECRET", "")
        capability_name = capability_name or os.environ.get("CAPABILITY_NAME", "pytrickle-worker")
        capability_desc = capability_desc or os.environ.get("CAPABILITY_DESCRIPTION", "PyTrickle video processing worker")
        capability_url = os.environ.get("CAPABILITY_URL", "http://localhost:8000")
        capability_capacity = capability_capacity or int(os.environ.get("CAPABILITY_CAPACITY", 1))
        capability_price_per_unit = capability_price_per_unit or int(os.environ.get("CAPABILITY_PRICE_PER_UNIT", 0))
        capability_price_scaling = capability_price_scaling or int(os.environ.get("CAPABILITY_PRICE_SCALING", 1))
        
        values = {
            "orch_url": orch_url,
            "orch_secret": orch_secret,
            "capability_name": capability_name,
            "capability_desc": capability_desc,
            "capability_url": capability_url,
            "capability_capacity": capability_capacity,
            "capability_price_per_unit": capability_price_per_unit,
            "capability_price_scaling": capability_price_scaling,
        }
        
        if not (values["orch_url"] and values["orch_secret"]):
            self.logger.info("Orchestrator URL or secret not provided, skipping capability registration")
            return False
            
        register_req = self._build_register_request(**values)
        try:
            result = await self._make_registration_request(
                values["orch_url"], 
                values["orch_secret"], 
                register_req, 
                max_retries, 
                delay, 
                timeout
            )
            return result
        finally:
            # Clear secret from memory
            if "orch_secret" in values:
                values["orch_secret"] = None
            orch_secret = None
    
    @classmethod
    async def register(cls, logger: Optional[logging.Logger] = None, **kwargs) -> Union[ParseResult, bool]:
        """
        Class method for simple one-line registration.
        
        Usage:
            RegisterCapability.register()  # Uses all env vars
            RegisterCapability.register(capability_name="my-worker")  # Custom name, env vars for rest
        
        Args:
            logger: Optional logger instance
            **kwargs: Any parameters to override (same as register_capability, except capability_url which always comes from CAPABILITY_URL env var)
            
        Returns:
            The registered capability URL (as ParseResult object) if registration succeeded, False otherwise
        """
        instance = cls(logger)
        return await instance.register_capability(**kwargs)
