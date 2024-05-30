import aiohttp

async def async_post(url, data, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=headers) as response:
            response.raise_for_status()  # Raise an exception for non-2xx status codes
            return response
        
async def async_get(url, data, headers):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an exception for non-2xx status codes
            return response