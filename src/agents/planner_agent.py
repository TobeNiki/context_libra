from pydantic import BaseModel
from agents import Agent


PROMPT = (

)

class StoryPlotPlan(BaseModel):
    pass 


story_plot_planner_agent = Agent(
    name="StoryPlotPlannerAgent",
    instructions=PROMPT,
    model="o3-mini",
    output_type=StoryPlotPlan,
)