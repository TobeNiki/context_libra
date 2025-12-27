import asyncio
from src.agents.scene_evaluation_agent import scene_evaluation
r = asyncio.run( scene_evaluation("./test_chunk4.txt"))
print(r)