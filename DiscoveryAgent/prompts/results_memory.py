CONTINUE_GEN_PROMPT = """
Be sure to integrate the report from the previous work on this dataset and task that you were given back at the beginning of our conversation.
Carry over information from this report to the new report, and make sure to not lose any critical details."""

CONTINUATION_REPORT_PROMPT = """You're now tasked with writing a report on all that you've done so far.
Your goal in this report should be to convey an honest, open, and full summary of your findings.
Make sure to include all your findings during our conversation, even ones that we may have talked about back at the beginning.
However, if you have any results or findings that are redundant, only include them once.
If you have any findings that aren't relevant to this task, do not include them.
If any of your results are contradictory, make sure to include only the one you're most confident in based on the data.

This report will be built on by scientists and researchers in the future. 
The focus here is not delivering a scientific finding, it's on conveying an honest interpretation of the data and any methods you came up with.
Do not try to persuade, but focus on what's scientifically valuable to the research community.

You should structure it like this:

# Background and Literature Review
- This section contains background information about the dataset and a literature review or searching you've done on it.
- Focus on including the most relevant papers or other sources that are useful to understand this dataset and the methods you've used.
- Use citations in your writing. Use [1], [2], [3], etc. to reference the citations in the text, and copy the citations, along with their numbers, into a references section (described below).
- In the final section, titled "### References", you should include exact citations. IMPORTANT: Copy the citations exactly from the citations returned by your helper agents. DO NOT INVENT OR ATTEMPT TO INFER CITATIONS FROM THE TEXT.
 
# Data Characteristics
- This section should convey a comprehensive view of the data and it's characteristics, including important statistics or qualities that should be taken into consideration when using it.
- Include both new analyses you ran on the data and overall characteristics that are already known (through the background provided to you, for example).

# Experiments and Analyses
- You should put all of your experiments and analyses into this section, making sure to structure them in a logical and coherent way.
- Format it with sub-sections with this exact format:

## [Experiment Name]
Hypothesis: [The original hypothesis you sought to test]
Methods: [Any methodologies you used to test the hypothesis]
Results: [The results of your experiment. Was the hypothesis proven or disproven? Were there any additional insights or findings from this experiment?]
Figures: [Any figures or images that are relevant to the experiment and results]
- For figures, you should use standard markdown image syntax to include them in the report. Use the `![Figure Description](path/to/figure)` syntax. IMPORTANT: Only include figures that are explicitly present in the codebase. Write the file paths EXACTLY or your images will not be included. DO NOT INVENT OR GUESS FIGURE FILENAMES.

# Discussion
- This section should summarize your overall findings. Focus on things that are interesting and scientifically valuable.
- What oddities or surprising findings should I know about this dataset? What did you find that was most surprising?
- What are avenues for future research or investigation? If you had more time, what would you look into or what experiments would you try?
- What should I or other researchers know about your work if we want to extend or evaluate it? Focus on limitations and caveats. You should be honest, transparent, and straightforward.

Think hard before writing the report as it's very important that I have clear, correct, and comprehensive information.
"""