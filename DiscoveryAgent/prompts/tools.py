# ------------------------------------- TOOL PROMPTS -------------------------------------
SINGLE_WEB_SEARCH_TAG = "web_search"
SINGLE_WEB_SEARCH_PROMPT = """
Description: Performs a single web search for the given query.
Guidelines: Use for quick searches that don't require deep synthesis of literature.
    - Useful for fast queries about facts, databases, or specific literature.
Output: A short summary of the search results.
"""
WEB_SEARCH_PROMPT = """
Description: Performs a web search for the given query.
Guidelines: Use for gathering information from the web. 
    - Uses an agent that has access to the internet to search publications and other resources related to your task.
    - Use this agent to gather up-to-date information on types of analyses or related information about the data to guide your hypothesis generation and experimentation.
Output: A report that summarizes the search results.
"""

DEEP_RESEARCH_TAG = "deep_research"
DEEP_RESEARCH_PROMPT = """
Description: A deep research agent that can perform massive search of literature and synthesis of such literature into a concise report.
Guidelines: Use for comprehensive literature analyses and massive search of resources.
    - Searches can be broad but try to remain focused on tractible areas of literature.
    - Deep research queries are expensive, so use them sparingly.
    - Useful at the start of your experimentation to get a sense of interesting directions to explore within the data.
Output: A report that synthesizes literature and research based on your query.
"""

VALENCY_AGENT_TAG = "valency_agent"
VALENCY_AGENT_PROMPT = """
Description: A scholarly literature agent powered by Valency Bond (mcp.valency.io) that searches 44M+ scientific preprints and peer-reviewed papers.
Guidelines: Use for precise academic citations, citation-graph traversal, and tracking recent preprints.
    - Useful for investigating methodology, finding related prior works, and resolving literature contradictions.
    - Queries return source-linked, verifiable papers with titles, authors, and DOIs/arXiv IDs.
Output: A source-grounded academic literature synthesis report with full bibliographic metadata.
"""

EXECUTION_AGENT_TAG = "execution_agent"
EXECUTION_AGENT_PROMPT = """
Description: An agent that is able to execute your queries through code and experimentation on the dataset.
Guidelines: Use for specific, actionable queries that will lead to meaningful insights.
    - Queries should be explicit, stating the experiments and analyses you would like to see on the dataset.
    - Some steps can be left up to interpretation as the agent can execute and edit code iteratively. You don't need to describe how to exactly implement every step of the analysis you request.
    - The execution agent is instructed to not execute analyses outside of your query, so try to comprehensively describe what you want the agent to do.
    - The agent is a terminal coding agent with full access to execution, editing, and reading capabilities. Use best practices for querying these agents.
    - The execution agent only has a limited budget of queries, so be efficient in what you ask for it.
    - Give the agent contained, focused tasks. If you need to test multiple analyses, do this across multiple queries rather than giving multiple distinct analyses in one query.
    - While the execution_agent can solve some problems on its own, consider giving it literature-supported advice or information in order to guide its implementation. This will allow it to be more efficient and rigorous as it cannot perform its own literature searches.
Output: A report that summarizes findings as are relevant to your query. The report covers the methodology used to execute your query as well as a comprehensive coverage of results.
"""