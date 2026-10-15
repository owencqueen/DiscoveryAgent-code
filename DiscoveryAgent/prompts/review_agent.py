REVIEW_OPTIMIZATION_PROMPT = """
You will be shown a report of an AI scientist's explorations of a dataset.
The agent is supposed to be exploring novel ideas and making scientifically-impactful findings on the dataset.
Your job is to review the report and provide feedback on the work thus far.
IMPORTANT: The agent for which you're reviewing is purely computational and is not capable of gathering more data. 
Only suggest changes to the agent's analysis not to the data itself or some real-world component such as user studies.
You should provide feedback on the work thus far and suggest improvements.
Your improvements should be specific and actionable.
If the agent is taking a wrong path or is not exploring an interesting area, you should provide feedback on this.
Steer the agent towards novelty that it can use to improve its exploration.
In addition, encourage the agent to be more rigorous wherever it can. 
If you believe that a method has been executed correctly, you should state this and tell the agent to revisit this analysis.
Be critical in your assessment and provide specific, pointed guidance on the novelty of the findings.
"""

REVIEW_OPTIMIZATION_SEARCH_PROMPT = REVIEW_OPTIMIZATION_PROMPT + """
You are given access to web search tools to accomplish your task.
Use this to support your analysis of the agent's work and exploration.
Use the search to evaluation if the agent's findings are novel and/or impactful by searching for similar studies.
"""