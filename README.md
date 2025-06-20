# Group Project Repository

This repository contains all materials for the group project in the course **WI4450: Special Topics in Computational Science and Engineering (2024/2025 Q3–Q4)**. The structure is designed to organize deliverables clearly and support reproducibility and collaboration.

## Directory Structure

### `data/`
- Contains all data required to run the code (if applicable).
- This may include raw datasets, preprocessed inputs, or synthetic data generated for testing.

### `code/`
- Contains a Jupyter notebook named `main.ipynb` with *all your* code needed to reproduce the results in the report.
- If third-party Python package source code is required, include it as subdirectories within the `code/` folder.
- The notebook should be:
  - Readable and well-structured
  - Easy to run and understand
  - Self-contained, with clearly defined inputs and outputs
- Theoretical derivations or detailed explanations should be included in the report. However, if such material is necessary to follow the notebook or understand the experimental reasoning, it should also be included in the notebook.

### `presentation/`
- Contains the slides used in the final project presentation.
- You may use LaTeX (beamer), PowerPoint, or another format. LaTeX (beamer) is recommended.
- Instructions for the Presentations
  - Each group has **35 minutes total**, including:
    - **20–25 minutes for the presentation**
    - **10–15 minutes for questions**
    - **Do not exceed 25 minutes** for the presentation itself
  - The presentation should:
    - Be **comprehensible to all course participants**
    - Have a **clear and logical structure** (the structure of the report can be a guideline)
    - **Tell a compelling story** to engage the audience
    - Prioritize a **clear narrative over completeness**; you do not need to show all results, as these will be included in the report and submitted code/notebooks
  - **All group members must actively contribute** to:
    - The **presentation**
    - The **question and answer session**
  - The presentation should include:
    - A **clear motivation and statement of your research question**
    - References to **relevant state-of-the-art literature**
    - An explanation of your **methodology and results**
    - An **answer to your research question**, or a discussion of what your findings reveal about it
    - (Optional) **Live code demonstration**
    - (Optional) **Suggestions for future work**
  - Grading information:
    - **All of the above aspects will be considered**
    - The presentation accounts for **50% of the final grade**
- **Additional instructions**:
  - Upload your **slides and/or notebook** to the group repository **before June 18**
  - There will be a **10-minute break after every second presentation**
  - If you plan to use your **own laptop** (e.g., for code demonstrations), please **test it before the session starts** or during the break before your slot
  - Unless you have a **conflicting important appointment or course**, please **stay for the entire session** to support your peers

### `report/`
- Contains the LaTeX source files for the project report.
- Use the provided template: [https://dzwaneveld.github.io/](https://dzwaneveld.github.io/) for consistency.
- The main text of the report should not exceed **10 pages** (excluding title page, TOC, references, etc.).
- The report should be concise yet self-contained. Concepts and methods not covered in lectures must be clearly explained.
- A typical structure includes:
  - **Introduction**: Introduce the problem and its relevance.
  - **Literature Review**: Summarize related work and highlight open questions.
  - **Research Question**: Clearly state the research question or hypothesis.
  - **Methodology**: Describe the approach, algorithms, and methods used.
  - **Numerical Results**: Present and analyze your numerical results or simulations.
  - **Discussion and Conclusion**: Interpret the findings, discuss implications, and suggest possible future work.
- Be selective with your results—only include those relevant to your discussion.
- Include all necessary files to compile the report (e.g., `.tex`, `.bib`, figures, custom style files).

### `project-proposal/`
- Contains the initial project proposal.
- The proposal should include:
  - A brief motivation for the selected topic
  - A clearly defined research question
  - A few key references (literature and/or code)

## Instructions

- All code must be written in **Python**.
- Include a `requirements.txt` file in the root directory listing all required packages.
- If needed, include external Git repositories as submodules using `git submodule add`. See: [https://git-scm.com/book/en/v2/Git-Tools-Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- Clearly disclose the use of AI tools (e.g., for writing code or text) in the report and/or notebook.
- Cite all references and give appropriate credit for any external code, libraries, or resources used.
- All original code and numerical experiments must be implemented in the Jupyter notebook.
- The notebook should provide sufficient description of the numerical experiments. Do not repeat all experiment details in the report—only include what's necessary for understanding the discussion.
- Work directly in this repository (or in submodules) to ensure all changes are properly tracked and reproducible.
- Maintain a working version of the report and update it regularly throughout the project, rather than completing it only at the end.